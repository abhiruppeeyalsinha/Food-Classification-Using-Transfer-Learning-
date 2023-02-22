import os, cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import load_img,img_to_array

model_path = 'model_file.h5'
model = load_model(model_path)
# print(model.summary())


y_true = []
y_pred = []
img_path = r"E:\Projects & Tutorial\CNN Project\unsupervised learning\food recog\Food-5K\sub_folder\evaluation"
for i in os.listdir(img_path):
    img = cv2.imread(img_path+"/"+i)
    disp_img = img.copy()
    img = cv2.resize(img, (225,225))
    img = img/255.
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    # y_true.append(int(i.split('_')[0]))
    # y_pred.append(1 if model.predict(img)>0.5 else 0)
    result = model.predict(img)
    val = (np.amax(result))
    val = round(val*100,2)
    print(f"Accuracy- {val}%")
    if result>0.5:
        print(1)
    else:
        print(0)
    cv2.imshow(str(i), disp_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()



