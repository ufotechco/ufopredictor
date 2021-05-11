from keras.preprocessing import image
from keras.models import load_model, model_from_json
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors
import keras.backend as K
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import sys
import uuid

class UFOPredictor:
    octType = None
    originalData = None
    inputData = None
    outputData = None

    availableModels = []
    selectedModel = None
    loadedModel = None
    threshold = 0.95

    temporalImageName = "/tmp/{}.png".format(uuid.uuid1())

    predictionResultMessage = []

    def __init__(self):
        pass

    def _loadPredictor(self):
        if self.selectedModel:
            json_file = open(self.selectedModel["model"], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.loadedModel = model_from_json(loaded_model_json)
            self.loadedModel.load_weights(self.selectedModel["weights"])

    def setPredictor(self, labelTag, threshold=0.95):
        for i, m in enumerate(self.availableModels):
            if m["label"] == labelTag:
                # Founded
                self.selectedModel = self.availableModels[i]
                self.threshold = threshold
        if self.selectedModel == None:
            raise NameError("El modelo no ha sido encontrado en las opciones disponibles que son {}".format([m.label for i in self.availableModels].join(',')))
        else:
            self._loadPredictor()
    def addModelPaths(self, labelTag, modelFilePath, weightsFilePath, predictionMsg):
        if labelTag != None and modelFilePath != None and weightsFilePath != None:
            self.availableModels.append({
                'label': labelTag,
                'model': modelFilePath,
                'weights': weightsFilePath,
                'predictionMsg': predictionMsg
            })
        else:
            raise NameError("Las variables no se encuentran bien diligenciadas. Recuerde colocar la ruta del modelo y la ruta de los pesos del modelo")

    
    def _comp_loss(self):
        preds = self.loadedModel.predict(self.inputData)
        prob = round(np.amax(preds),2)
        ind = preds.argmax()
        return prob, ind

    def _processImage(self):
        imgPath = self.inputData
        im = Image.open(imgPath)
        im = ImageEnhance.Brightness(im).enhance(0.7)
        im = ImageEnhance.Contrast(im).enhance(2.0)
        im.thumbnail((299, 299), Image.ANTIALIAS)
        im.save(self.temporalImageName)
        imagePath = self.temporalImageName
        img_path = imagePath
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x /= 255.
        x = np.expand_dims(x, axis=0)
        self.inputData = x

    def _grad_cam(self):
        model = self.loadedModel
        preds = model.predict(self.inputData)
        class_idx = np.argmax(preds[0])
        class_output = model.output[:, class_idx]
        last_conv_layer = model.get_layer('conv2d_94')
        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([self.inputData])
        for i in range(192):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        img = cv2.imread(self.originalData)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        probabilidad = round(np.amax(preds)*100,2)
        if self.outputData:
            if ".png" in self.outputData or ".jpg" in self.outputData:
                cv2.imwrite(self.outputData, superimposed_img)
            else:
                raise NameError("La variable de salida debe corresponder a una ruta de imagen")


    def process(self, inputPath, outputPath):
        self.inputData = inputPath
        self.originalData = inputPath
        self.outputData = outputPath
        self.isResult = False
        self._processImage()
        probaResult, indResult = self._comp_loss()
        self._grad_cam()

        if probaResult >= self.threshold and indResult == 0:
            self.predictionResultMessage.append(self.selectedModel["predictionMsg"])
            self.isResult = True
        return self.isResult