import Augmentor
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def gen_red_generators(train_dir, val_dir, test_dir, input_size=(316,316,3), batch_size=10,
                       class_mode='categorical', red_prob=0.4, red_grid=[3,3], red_magnitude=4):
    
    if red_prob == 0:
        train_datagen = ImageDataGenerator(rescale=1./255)
    else:
        p = Augmentor.Pipeline()
        p.random_distortion(probability=red_prob, grid_width=red_grid[0], grid_height=red_grid[1],
                                magnitude=red_magnitude)
        train_datagen = ImageDataGenerator(preprocessing_function=p.keras_preprocess_func())
    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=input_size,
            batch_size=batch_size,    
            class_mode=class_mode)

    validation_generator = test_datagen.flow_from_directory(
            val_dir,
            target_size=input_size,
            batch_size=batch_size,
            class_mode=class_mode)

    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=input_size,
            batch_size=batch_size,
            class_mode=class_mode)

    return train_generator, validation_generator, test_generator