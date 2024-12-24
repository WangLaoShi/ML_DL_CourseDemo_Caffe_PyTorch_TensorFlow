import tensorflow as tf
'''
This code imports the dataset and net modules, takes in a text file as an argument, 
sets up the batch size, number of classes, image size, and learning rate, and then 
creates a dataset object. It then creates an iterator to iterate through the dataset, 
calculates the logits of the images, applies softmax to the logits, calculates the cross 
entropy, accuracy, and train step, and then saves the model every 100 iterations. 

If debug is set to true, it will also display the image and label for the first image in the batch.
'''

batch_size    = 16
num_classes   = 2
image_size    = (48,48)
img_height    = 48
img_width     = 48
learning_rate = 0.0001

AUTOTUNE = 1000

path = "../../../../datas/mouth/"

if __name__=="__main__":
    """
    What is image_dataset_from_directory?Permalink
    image_dataset_from_directory generates a a tf.data.Dataset from image files in a directory that yield batches of images from the sub-directories or class directory within the main directory and labels that is the sub-directory name
    
    tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        **kwargs
    )
    Arguments:
    
    directory: Image Data path. Ensure it contains sub-directories of Image class if labels is set to “inferred”
    
    labels: default is inferred (labels are generated from sub-directories of Image classes or a list/tuple of integer labels of same size as number of images in the directory label
    
    label_mode: int - if labels are integers sparse_categorical_crossentropy loss categorical - labels are categorical for categorical_crossentropy loss binary - labels are either 0 or 1 for binary_crossentropy
    
    class_names: Only if “labels” is “inferred” and is used for the order of classes otherwise it’s sorted in alphanumeric order
    
    color_mode: image channels. default is ‘rgb’, if grayscale then it’s 1
    
    batch_size: Image batch size, default is 32
    
    image_size: default is (256,256)
    
    shuffle: data shuffle, Boolean, default is True
    
    validation_split: percent of data reserve for validation
    
    subset: “training” for training data and “validation” for test data. Only used if validation_split is set.
    
    Interpolation: default is bilinear and used when resizing image
    
    crop_to_aspect_ratio: True or False, If set to true then resize image without aspect ratio distortion
    
    Returns
    
    A tf.data.Dataset object
    """

    # https://kanoki.org/2022/06/09/image-dataset-from-directory-in-tensorflow/

    # Found 1000 files belonging to 2 classes.
    # Using 800 files for training.
    # <BatchDataset element_spec=(TensorSpec(shape=(None, 48, 48, 3),
    # dtype=tf.float32, name=None),
    # TensorSpec(shape=(None,), dtype=tf.int32, name=None))>
    train_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    print(train_ds)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    print(val_ds)


    ## Demo the picture
    class_names = train_ds.class_names
    print(class_names)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch)
        break

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))

    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()
    ## Demo the picture

    print('Number of training batches: %d' % tf.data.experimental.cardinality(train_ds).numpy())
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds).numpy())

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    DataSet
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = val_ds.prefetch(buffer_size=AUTOTUNE)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    : Model Initialize
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(img_width, img_height, 3)),
        tf.keras.layers.Reshape((img_width, img_height * 3)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(img_width, return_sequences=True, return_state=False)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(img_width)),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(256),

    ])

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10))
    model.summary()

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    : Optimizer
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
        name='Nadam'
    )  # 0.00001

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    : Loss Fn
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # 1
    # lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_logarithmic_error')

    # 2
    lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.AUTO,
                                                           name='sparse_categorical_crossentropy')

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    : Model Summary
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy()])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    : Training
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # history = model.fit(train_dataset, epochs=15000, validation_data=(validation_dataset))
    history = model.fit(train_dataset, epochs=15, validation_data=(validation_dataset))

    # evaluate the model
    # scores = model.evaluate(X, Y)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    print("Layer0 Type->",type(model.layers[0]))

    from ann_visualizer.visualize import ann_viz;

    ann_viz(model, title="My first neural network")

    input("Press Any Key!")

