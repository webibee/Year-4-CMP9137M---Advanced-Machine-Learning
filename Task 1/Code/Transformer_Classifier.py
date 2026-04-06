# This code is built on top of the ITM-CNN classifier as provided by Heriberto Cuayahuitl, University of Lincoln. 
# Contact: hcuayahuitl@lincoln.ac.uk
#####################################################

# Let's import the dependencies
import sys
import os
import time
import einops
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from official.nlp import optimization
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

mixed_precision.set_global_policy('mixed_float16')

# Class for loading image and text data
class ITM_DataLoader():
    BATCH_SIZE = 16
    IMAGE_SIZE = (224, 224)
    IMAGE_SHAPE = (224, 224, 3)
    SENTENCE_EMBEDDING_SHAPE = (384)
    AUTOTUNE = tf.data.AUTOTUNE
    IMAGES_PATH = "C:\\Users\\benwe\\Desktop\\flickr8k.dataset-cmp9137-item1\\flickr8k-resised"
    train_data_file = IMAGES_PATH+"\\..\\flickr8k.TrainImages.txt"
    dev_data_file = IMAGES_PATH+"\\..\\flickr8k.DevImages.txt"
    test_data_file = IMAGES_PATH+"\\..\\flickr8k.TestImages.txt"
    sentence_embeddings_file = IMAGES_PATH+"/../flickr8k.cmp9137.sentence_transformers.pkl"
    sentence_embeddings = {}
    train_ds = None
    val_ds = None
    test_ds = None

    def __init__(self):
        self.sentence_embeddings = self.load_sentence_embeddings()
        self.train_ds = self.load_classifier_data(self.train_data_file)
        self.val_ds = self.load_classifier_data(self.dev_data_file)
        self.test_ds = self.load_classifier_data(self.test_data_file)
        print("done loading data...")

    # Sentence embeddings are dense vectors representing text data, one vector per sentence. 
    # Sentences with similar vectors would mean sentences with equivalent meanning.  
	# They are useful here to provide text-based features of questions in the data.
    # Note: sentence embeddings don't include label info, they are solely based on captions.
    def load_sentence_embeddings(self):
        sentence_embeddings = {}
        print("READING sentence embeddings...")
        with open(self.sentence_embeddings_file, 'rb') as f:
	        data = pickle.load(f)
	        for sentence, dense_vector in data.items():
		        #print("*sentence=",sentence)
		        sentence_embeddings[sentence] = dense_vector
        print("Done reading sentence_embeddings!")
        return sentence_embeddings

    # In contrast to text-data based on pre-trained features, image data does not use
    # any form of pre-training in this program. Instead, it makes use of raw pixels.
    # Notes that input features to the classifier are only pixels and sentence embeddings.
    def process_input(self, img_path, dense_vector, text, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.IMAGE_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.cast(img, tf.float32) / 255
        features = {}
        features["image_input"] = img
        features["text_embedding"] = dense_vector
        features["caption"] = text
        features["file_name"] = img_path
        return features, label

    # This method loads the multimodal data, which comes from the following sources:
    # (1) image files in IMAGES_PATH, and (2) files with pattern flickr8k.*Images.txt
    # The data is stored in a tensorflow data structure to make it easy to use by
    # the tensorflow model during training, validation and test. This method was 
    # carefully prepared to load the data rapidly, i.e., by loading already created
    # sentence embeddings (text features) rather than creating them at runtime.
    def load_classifier_data(self, data_files):
        print("LOADING data from "+str(data_files))
        print("=========================================")
        image_data = []
        text_data = []
        embeddings_data = []
        label_data = []
		
        # get image, text, label of image_files
        with open(data_files) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("	")
                img_name = os.path.join(self.IMAGES_PATH, img_name.strip())

                # get binary labels from match/no-match answers
                label = [1, 0] if raw_label == "match" else [0, 1]
                #print("I=%s T=%s _L=%s L=%s" % (img_name, text, raw_label, label)) 

				# get sentence embeddings (of textual captions)
                text_sentence_embedding = self.sentence_embeddings[text]
                text_sentence_embedding = tf.constant(text_sentence_embedding)

                image_data.append(img_name)
                embeddings_data.append(text_sentence_embedding)
                text_data.append(text)
                label_data.append(label)

        print("|image_data|="+str(len(image_data)))
        print("|text_data|="+str(len(text_data)))
        print("|label_data|="+str(len(label_data)))
		
        # prepare a tensorflow dataset using the lists generated above
        dataset = tf.data.Dataset.from_tensor_slices((image_data, embeddings_data, text_data, label_data))
        dataset = dataset.shuffle(self.BATCH_SIZE * 8)
        dataset = dataset.map(self.process_input, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.batch(self.BATCH_SIZE).prefetch(self.AUTOTUNE)
        self.print_data_samples(dataset)
        return dataset

    def print_data_samples(self, dataset):
        print("PRINTING data samples...")
        print("-----------------------------------------")
        for features_batch, label_batch in dataset.take(1):
            for i in range(1):
                print(f'Image pixels: {features_batch["image_input"]}')
                print(f'Sentence embeddings: {features_batch["text_embedding"]}')
                print(f'Caption: {features_batch["caption"].numpy()}')
                label = label_batch.numpy()[i]
                print(f'Label : {label}')
        print("-----------------------------------------")



# Main class for the Image-Text Matching (ITM) task
class Transformer_Classifier(ITM_DataLoader):
    epochs = 10
    learning_rate = 3e-5
    class_names = {'match', 'no-match'}
    num_classes = len(class_names)
    classifier_model = None
    history = None
    classifier_model_name = 'ITM_Classifier-flickr'
    MAX_IMG_LENGTH = 225
    MAX_TEXT_LENGTH = 512 # maximum sequence length of 512 tokens
    use_auth_token=True

    def __init__(self):
        super().__init__()
        self.build_classifier_model()
        self.train_classifier_model()
        self.test_classifier_model()

    def mlp(self, transformer_layer, hidden_units, dropout_rate):
        for units in hidden_units:
            transformer_layer = keras.layers.Dense(units, activation=tf.nn.gelu)(transformer_layer)
            transformer_layer = keras.layers.Dropout(dropout_rate)(transformer_layer)
        return keras.layers.Dense(hidden_units[-1])(transformer_layer)
    
    def create_vision_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        img_input = layers.Input(shape=self.IMAGE_SHAPE, name="image_input")
        transformer_layer = layers.Rescaling(1.0/255)(img_input)
        transformer_layer = layers.Conv2D(12, kernel_size=4, strides=2, padding='same', activation='relu')(transformer_layer)
        transformer_layer = layers.MaxPooling2D(pool_size=4, strides=2, padding='same')(transformer_layer)
        transformer_layer = layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu')(transformer_layer)
        transformer_layer = layers.MaxPooling2D(pool_size=4, strides=2, padding='same')(transformer_layer)
        transformer_layer = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')(transformer_layer)
        transformer_layer = layers.MaxPooling2D(pool_size=4, strides=2, padding='same')(transformer_layer)

        for _ in range(num_projection_layers):
            x1 = keras.layers.LayerNormalization(epsilon=1e-6)(transformer_layer)
            attention_output = keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(x1, x1)
            x2 = keras.layers.Add()([attention_output, transformer_layer])
            x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self.mlp(x3, hidden_units=[128,64], dropout_rate=0.1)
            transformer_layer = keras.layers.Add()([x3, x2])

        representation = keras.layers.LayerNormalization(epsilon=1e-6)(transformer_layer)
        representation = keras.layers.Flatten()(representation)
        representation = keras.layers.Dropout(0.5)(representation)
        outputs = self.mlp(representation, hidden_units=[256,128], dropout_rate=0.5)
        return img_input, outputs
    
    # return learnt feature representations based on dense layers, dropout, and layer normalisation
    def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = layers.Dense(projection_dims)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings
    
    # return learnt feature representations of input data (text embeddings in the form of dense vectors)
    def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        text_input = keras.Input(shape=self.SENTENCE_EMBEDDING_SHAPE, name='text_embedding')
        outputs = self.project_embeddings(text_input, num_projection_layers, projection_dims, dropout_rate)
        return text_input, outputs

    def build_classifier_model(self):
        print(f'BUILDING model')
        img_input_ids, img_encoder_output = self.create_vision_encoder(num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        text_input_ids, text_encoder_output = self.create_text_encoder(num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        net = tf.keras.layers.Concatenate(axis=1)([img_encoder_output, text_encoder_output])
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(self.num_classes, activation='softmax', name=self.classifier_model_name)(net)
        self.classifier_model = tf.keras.Model(inputs=[img_input_ids, text_input_ids], outputs=net)
        self.classifier_model.summary()

    def train_classifier_model(self):
        print(f'TRAINING model')
        steps_per_epoch = tf.data.experimental.cardinality(self.train_ds).numpy()
        num_train_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(0.2*num_train_steps)

        loss = tf.keras.losses.KLDivergence()
        metrics = tf.keras.metrics.BinaryAccuracy()
        optimizer = optimization.create_optimizer(init_lr=self.learning_rate,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

        self.classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.history = self.classifier_model.fit(x=self.train_ds, validation_data=self.val_ds, epochs=self.epochs)
        print("model trained!")

    def test_classifier_model(self):
        print("TESTING classifier model (showing a sample of image-text-matching predictions)...")
        num_classifications = 0
        num_correct_predictions = 0
        y_true = []
        y_pred = []

        # read test data for ITM classification
        for features, groundtruth in self.test_ds:
            groundtruth = groundtruth.numpy()
            predictions = self.classifier_model(features)
            predictions = predictions.numpy()
            captions = features["caption"].numpy()
            file_names = features["file_name"].numpy()

            # read test data per batch
            for batch_index in range(0, len(groundtruth)):
                predicted_values = predictions[batch_index]
                probability_match = predicted_values[0]
                probability_nomatch = predicted_values[1]
                predicted_class = "[1 0]" if probability_match > probability_nomatch else "[0 1]"
                y_pred.append(predicted_class)
                y_true.append(np.argmax(groundtruth[batch_index]))
                if str(groundtruth[batch_index]) == predicted_class: 
                    num_correct_predictions += 1
                num_classifications += 1

                # print a sample of predictions -- about 10% of all possible
                if random.random() < 0.1:
                    caption = captions[batch_index]
                    file_name = file_names[batch_index].decode("utf-8")
                    print("ITM=%s PREDICTIONS: match=%s, no-match=%s \t -> \t %s" % (caption, probability_match, probability_nomatch, file_name))
        
        # Convert string labels to numerical labels for y_pred
        y_pred_numeric = [1 if label == "[1 0]" else 0 for label in y_pred ]

        # reveal test performance using our own calculations above
        accuracy = num_correct_predictions/num_classifications
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred_numeric)
        precision = precision_score(y_true, y_pred_numeric)
        recall = recall_score(y_true, y_pred_numeric)
        f1 = f1_score(y_true, y_pred_numeric)

        print("TEST accuracy: %4f" % (accuracy))
        print("Balanced Classification Accuracy: %4f" % (balanced_accuracy))
        print("Precision: %4f" % (precision))
        print("Recall: %4f" % (recall))
        print("F1-score: %4f" % (f1))

        # reveal test performance using Tensorflow calculations
        loss, accuracy = self.classifier_model.evaluate(self.test_ds)
        print(f'Tensorflow test method: Loss: {loss}; ACCURACY: {accuracy}')



# Let's create an instance of the main class
itm = Transformer_Classifier()



























    #def create_transformer_encoder(self, input_shape, model_name):
        #input_ids = tf.keras.layers.Input(shape=input_shape, dtype=tf.int32, name="input_ids")
        #tokenizer = BertTokenizer.from_pretrained(model_name)
        #transformer_model = TFBertModel.from_pretrained(model_name)
        #encoder_output = transformer_model(input_ids)[0]
        #return input_ids, encoder_output
    
    #def encode_image(self, img_input_ids):
     #   # Add batch dimension to the input tensor
      #  img_input_ids = tf.expand_dims(img_input_ids, axis=0)

        # Ensure input tensor has the correct shape (224, 224, 3)
#        img_input_ids = tf.image.resize(img_input_ids, (224, 224))

        # Duplicate the single channel to 3 channels
 #       img_input_ids = tf.tile(img_input_ids, [1, 1, 3])

        # Load the pre-trained ResNet model
  #      resnet_model = tf.keras.applications.ResNet50(
   #         include_top=False,  # Exclude the fully-connected layer at the top
    #        weights='imagenet',  # Use pre-trained ImageNet weights
     #       input_shape=(224, 224, 3)  # Input shape of images
 ###       )
#
        # Freeze the layers of the ResNet model
    #    for layer in resnet_model.layers:
     #       layer.trainable = False

        # Encode image input ids using the ResNet model
      #  img_encoder_output = resnet_model(img_input_ids)

       # return img_encoder_output



    
    #def encode_text(self, text_input_ids):
        # Load a pre-trained text encoder model - BERT
        #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        #text_encoder_model = TFBertModel.from_pretrained("bert-base-uncased")

        #Encode text input ids using the text encoder model
        #text_encoder_output = text_encoder_model(text_input_ids)[0]
        #return text_encoder_output

    # return learnt feature representations based on dense layers, dropout, and layer normalisation
    #def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        #projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        #for _ in range(num_projection_layers):
            #x = tf.nn.gelu(projected_embeddings)
            #x = layers.Dense(projection_dims)(x)
            #x = layers.Dropout(dropout_rate)(x)
            #x = layers.Add()([projected_embeddings, x])
            #projected_embeddings = layers.LayerNormalization()(x)
        #return projected_embeddings

    # return learnt feature representations of input data (text embeddings in the form of dense vectors)
    #def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        #text_input = keras.Input(shape=self.SENTENCE_EMBEDDING_SHAPE, name='text_embedding')
        #outputs = self.project_embeddings(text_input, num_projection_layers, projection_dims, dropout_rate)
        #return text_input, outputs

    # put together the feature representations above to create the image-text (multimodal) deep learning model
    #def build_classifier_model(self):
        #print(f'BUILDING model')
        #img_input, vision_net = self.create_vision_encoder(num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        #text_input, text_net = self.create_text_encoder(num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        #net = tf.keras.layers.Concatenate(axis=1)([vision_net, text_net])
        #net = tf.keras.layers.Dropout(0.1)(net)
        #net = tf.keras.layers.Dense(self.num_classes, activation='softmax', name=self.classifier_model_name)(net)
        #self.classifier_model = tf.keras.Model(inputs=[img_input, text_input], outputs=net)
        #self.classifier_model.summary()