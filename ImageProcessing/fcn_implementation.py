from torch import no_grad, argmax
import torch.nn as nn
import numpy as np
import cv2
import torchvision.models.segmentation as seg_models
from ImageProcessing.semantic_data import SegmentationSample
from ImageProcessing.ProcessAllSizes import MUtils


class SemanticSeg(nn.Module):
    def __init__(self):
        super(SemanticSeg, self).__init__()

        self.device = 'cpu'
        self.model = self.load_model(pretrained=True)

    def forward(self, input: SegmentationSample):
        with no_grad():
            output = self.model(input.processed_image)['out']
            # reshaped_output = argmax(output.squeeze(), dim=0).detach().cpu()
            reshaped_output = argmax(output.squeeze(), dim=0).detach().cpu().numpy()

        return reshaped_output

    # Add the Backbone option in the parameters
    def load_model(self, pretrained):
        # model = seg_models.deeplabv3_resnet101(pretrained)
        model = seg_models.fcn_resnet101(pretrained)

        model.to(self.device)
        model.eval()
        return model

    def run_bg_inference(self, image_foreground: SegmentationSample):
        model = SemanticSeg()
        output = model(image_foreground)
        new_img = self.remove_background(output, image_foreground.image_file)
        new_img = MUtils.image_resize(new_img, image_foreground.img_width, image_foreground.img_height, cv2.INTER_LINEAR)
        returnedIMG = MUtils.convert_image(new_img)
        return returnedIMG


    def remove_background(self, input_image, source, num_channels=21):
        # 0=background, 12=dog, 13=horse, 14=motorbike, 15=person
        label_colors = np.array([(0, 0, 0),
            (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
            (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
            (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
            (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(input_image).astype(np.uint8)
        g = np.zeros_like(input_image).astype(np.uint8)
        b = np.zeros_like(input_image).astype(np.uint8)

        # label 15 = person
        for l in range(0, num_channels):
            if l == 15:
                idx = input_image == l
                r[idx] = label_colors[l, 0]
                g[idx] = label_colors[l, 1]
                b[idx] = label_colors[l, 2]
            else:
                continue

        rgb = np.stack([r, g, b], axis=2)
        # return rgb

        # and resize image to match shape of R-band in RGB output map
        foreground = cv2.imread(source)
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
        foreground = cv2.resize(foreground, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        
        # Create a background array to hold white pixels
        background = 255 * np.ones_like(rgb).astype(np.uint8)

        # Convert uint8 to float
        foreground = foreground.astype(float)
        background = background.astype(float)

        # Create a binary mask of the RGB output map using the threshold value 0
        _, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)

        # Apply a slight blur to the mask to soften edges
        alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = alpha.astype(float)/255

        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha, foreground)

        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha, background)

        # Add the masked foreground and background
        outImage = cv2.add(foreground, background)

        return outImage / 255