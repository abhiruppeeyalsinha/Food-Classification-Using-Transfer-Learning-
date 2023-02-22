import cv2
import numpy as np, pandas as pd
import os
from keras.applications.resnet import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import Input, Model
import matplotlib.pyplot as plt


# for i, img_file in enumerate(os.listdir(path)):
#     img = cv2.imread(path+"/"+str(img_file))
#     img = cv2.resize(img, (550,550))
#     cv2.imwrite(
#         os.path.join(r"D:\Abhirup Galarus\smart hook model\hook img classification\dataset\crop images\dataset of pipes\new dataset",
#                      f"New_image- {i}" + ".JPEG"), img)
# path = r"D:\Abhirup Galarus\smart hook model\hook img classification\dataset\crop images\dataset of pipes\new dataset of pipe"

Food_5KPath ="Food-5K"
# print(os.listdir(Food_5KPath))

def dframe(datatype = " "):
    x = []
    y = []
    path = Food_5KPath +'/'+ datatype
    for i in os.listdir(path):
        x.append(i)
        y.append(i.split('_')[0])

    x = np.array(x)
    y = np.array(y)

    df = pd.DataFrame()
    df['filename'] = x
    df['labels'] = y
    return df

val = dframe(datatype='training')




data_train = dframe(datatype='training')
data_validation = dframe(datatype='validation')
data_evaluation = dframe(datatype='evaluation')


# print(data_train)
# print(data_validation)
# print(data_evaluation.head())

train_generator = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                     rotation_range=20,width_shift_range=0.2, height_shift_range=0.2,
                                     horizontal_flip=True)

val_datagenerator = ImageDataGenerator(featurewise_center = True,featurewise_std_normalization=True,
                                       rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                       horizontal_flip=True)

train_generator = train_generator.flow_from_dataframe(
    data_train,
    directory=r'E:\CNN Project\unsupervised learning\food recog\Food-5K\training',
    x_col='filename',
    y_col='labels',
    class_mode='binary',
    target_size=(225, 225)
)

validation_generator = val_datagenerator.flow_from_dataframe(
    data_train,
    directory=r'E:\CNN Project\unsupervised learning\food recog\Food-5K\validation',
    x_col='filename',
    y_col='labels',
    class_mode='binary',
    target_size=(225, 225)
)

featur_extractor = ResNet50(weights="imagenet", input_shape=(225,225,3), include_top=False)


featur_extractor.trainable = False
input_layer = Input(shape=(225,225,3))

x = featur_extractor(input_layer, training=False)
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(units=512, activation ='relu')(x)
x = Dropout(0.03)(x)
x = Dense(units=512, activation ='relu')(x)
x = Dropout(0.03)(x)
# x = featur_extractor(input_layer, training=False)(x)
# x = GlobalAveragePooling2D()(x)
output_data = Dense(1,activation='sigmoid')(x)
model = Model(input_layer, output_data)

model.compile(optimizer='adam', loss = "binary_crossentropy",metrics=["accuracy"])
# model.summary()

train_model=model.fit_generator(train_generator, epochs=20,validation_data= validation_generator, steps_per_epoch=len(train_generator),
                                validation_steps=len(validation_generator))

save_model = train_model.model.save(r"E:\CNN Project\unsupervised learning\food recog\model_file_20.h5")

plt.plot(train_model.history['loss'], label='train loss')
plt.plot(train_model.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(train_model.history['accuracy'], label='train acc')
plt.plot(train_model.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
