import uuid
import os
from PIL import Image
import numpy as np
from ImageProcessing.semantic_data import SegmentationSample
from ImageProcessing.fcn_implementation import SemanticSeg
from ImageProcessing.ProcessAllSizes import IMGSProcessor, AllMeasurements

def get_input_image_path(instance, filename):
    _, ext = os.path.splitext(filename)
    return 'media/Input_image/{}{}'.format(uuid.uuid4(), ext)


def get_output_image_path(instance, filename):
    _, ext = os.path.splitext(filename)
    return 'media/Output_image/{}{}'.format(uuid.uuid4(), ext)


def modify_input_for_multiple_files(property_id, image):
    dict = {}
    dict['property_id'] = property_id
    dict['image'] = image
    return dict


class RunSegmentationInference():
    def __init__(self, image_front, image_side):
        self.front_image = image_front
        self.side_image = image_side

        self.output_folder = 'media/Output_image/'

        self.front_base_path, self.front_filename = os.path.split(self.front_image.front_input_image.path)
        self.front_sample_image = SegmentationSample(root_dir = self.front_base_path, image_file=self.front_filename, device='cpu')
        
        self.side_base_path, self.side_filename = os.path.split(self.side_image.side_input_image.path)
        self.side_sample_image = SegmentationSample(root_dir = self.side_base_path, image_file = self.side_filename, device='cpu')

        self.model = SemanticSeg(pretrained=True, device='cpu')

        self.updated_front_image = None
        self.updated_side_image = None
        self.measurements = None


    def save_frontbg_output(self):
        res = self.model.run_bg_inference(self.front_sample_image)
        self.updated_front_image = res
        # image_to_array = Image.fromarray((res * 255).astype(np.uint8))
        image_to_array = Image.fromarray(res)
        image_to_array.save(self.output_folder + self.front_filename)
        self.front_image.output_image = self.output_folder + self.front_filename
        self.front_image.save()


    def save_sidebg_output(self):
        res = self.model.run_bg_inference(self.side_sample_image)
        self.updated_side_image = res

        image_to_array = Image.fromarray(res)
        image_to_array.save(self.output_folder + self.side_filename)
        self.side_image.output_image = self.output_folder + self.side_filename
        self.side_image.save()


    def process_imgs(self):
        front_img = self.updated_front_image
        side_img = self.updated_side_image

        img_processor = IMGSProcessor(front_img, side_img)
        frontMeasure, sideMeasure = img_processor.process_measurements()
        
        return AllMeasurements(frontMeasure, sideMeasure)