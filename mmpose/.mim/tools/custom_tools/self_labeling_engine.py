import os
import warnings
import json
import cv2
from imashrimp_ViTPose.mmpose.apis import (inference_top_down_pose_model, init_pose_model)
from imashrimp_ViTPose.mmpose.datasets import DatasetInfo
import zipfile
import shutil
from tools.custom_tools.old.test_tool_old import set_keypoints_circles
import numpy as np

try:
    from imashrimp_mmcv.mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from imashrimp_ViTPose.mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from imashrimp_ViTPose.mmpose.core import wrap_fp16_model


class SelfLabeling:
    def __init__(self, images_input_dir, images_output_dir, limit=None):
        self.images_input_dir = images_input_dir
        self.limit = limit
        self.images_output_dir = images_output_dir
        self.basename = "SelfLabeling" + os.path.basename(os.path.abspath(os.path.join(images_input_dir, ".")))
        self.img_out = os.path.join(self.images_output_dir, self.basename)
        # Directory to save results
        if not os.path.isdir(self.img_out):
            os.makedirs(self.img_out)
        # Directory data to save images and manifest.jsonl
        self.data_path = os.path.join(self.img_out, "data")
        if not os.path.isdir(os.path.join(self.img_out, "data")):
            os.makedirs(os.path.join(self.img_out, "data"))
        # Images list
        if "color_images" in os.listdir(os.path.join(self.images_input_dir)):
            self.image_path = os.path.join(self.images_input_dir, "color_images")
            self.img_list = os.listdir(os.path.join(self.images_input_dir, "color_images"))
            self.depth_path = os.path.join(self.images_input_dir, "depth_images")
            self.dpt_list = os.listdir(os.path.join(self.images_input_dir, "depth_images"))
        else:
            self.image_path = self.images_input_dir
            self.img_list = os.listdir(self.images_input_dir)
            self.depth_path = os.path.join(
                os.path.join(os.path.dirname(os.path.dirname(self.images_input_dir)), "depths"),
                os.path.basename(self.images_input_dir))
            self.dpt_list = os.listdir(self.depth_path)
        # Base Structure of annotation file
        stop_frame = (len(self.img_list) - 1) if limit is None else (limit - 1)
        self.annotation_json = [{"version": 0, "tags": [], "shapes": [], "tracks": [
            {"frame": 0, "group": 0, "source": "manual", "shapes": [
                {"type": "skeleton", "occluded": False, "outside": False, "z_order": 0, "rotation": 0.0, "points": [],
                 "frame": 0, "attributes": []}], "attributes": [],
             "elements": [{"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "1"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "2"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "3"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "4"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "5"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "6"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "7"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "8"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "9"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "10"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "11"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "12"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "13"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "14"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "15"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "16"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "17"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "18"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "19"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "20"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "21"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "22"},
                          {"frame": 0, "group": 0, "source": "manual", "shapes": [], "attributes": [], "label": "23"}],
             "label": "skeleton"}]}]
        # Base Structure of task file
        self.task_json = {'name': self.basename, 'bug_tracker': '', 'status': 'completed', 'subset': '',
                          'labels': [{'name': 'skeleton', 'color': '#91becf', 'attributes': [], 'type': 'skeleton',
                                      'sublabels': [
                                          {'name': '21', 'color': '#c0762a', 'attributes': [], 'type': 'points'},
                                          {'name': '22', 'color': '#76bb18', 'attributes': [], 'type': 'points'},
                                          {'name': '23', 'color': '#afc7d8', 'attributes': [], 'type': 'points'},
                                          {'name': '1', 'color': '#d12345', 'attributes': [], 'type': 'points'},
                                          {'name': '2', 'color': '#350dea', 'attributes': [], 'type': 'points'},
                                          {'name': '3', 'color': '#479ffe', 'attributes': [], 'type': 'points'},
                                          {'name': '4', 'color': '#4a649f', 'attributes': [], 'type': 'points'},
                                          {'name': '5', 'color': '#478144', 'attributes': [], 'type': 'points'},
                                          {'name': '6', 'color': '#57236b', 'attributes': [], 'type': 'points'},
                                          {'name': '7', 'color': '#1cdda5', 'attributes': [], 'type': 'points'},
                                          {'name': '8', 'color': '#e2bc6e', 'attributes': [], 'type': 'points'},
                                          {'name': '9', 'color': '#f067db', 'attributes': [], 'type': 'points'},
                                          {'name': '10', 'color': '#63bbfa', 'attributes': [], 'type': 'points'},
                                          {'name': '11', 'color': '#22b16f', 'attributes': [], 'type': 'points'},
                                          {'name': '12', 'color': '#daddec', 'attributes': [], 'type': 'points'},
                                          {'name': '13', 'color': '#2ac791', 'attributes': [], 'type': 'points'},
                                          {'name': '14', 'color': '#de22a0', 'attributes': [], 'type': 'points'},
                                          {'name': '15', 'color': '#a7a570', 'attributes': [], 'type': 'points'},
                                          {'name': '16', 'color': '#74db1b', 'attributes': [], 'type': 'points'},
                                          {'name': '17', 'color': '#aaa246', 'attributes': [], 'type': 'points'},
                                          {'name': '18', 'color': '#bfa552', 'attributes': [], 'type': 'points'},
                                          {'name': '19', 'color': '#9b8d37', 'attributes': [], 'type': 'points'},
                                          {'name': '20', 'color': '#c8023c', 'attributes': [], 'type': 'points'}],
                                      'svg': '<line x1="67.97002410888672" y1="35.34954833984375" x2="66.29774475097656" y2="49.73115158081055" data-type="edge" data-node-from="22" data-node-to="23"></line>/n<line x1="60.611995697021484" y1="37.857967376708984" x2="73.82300567626953" y2="41.871437072753906" data-type="edge" data-node-from="7" data-node-to="8"></line>\n<line x1="56.26407241821289" y1="35.85123062133789" x2="55.42793273925781" y2="49.229469299316406" data-type="edge" data-node-from="20" data-node-to="21"></line>\n<line x1="46.89930725097656" y1="32.841129302978516" x2="47.066532135009766" y2="48.39332962036133" data-type="edge" data-node-from="18" data-node-to="19"></line>\n<line x1="38.537906646728516" y1="32.841129302978516" x2="40.04296112060547" y2="47.055503845214844" data-type="edge" data-node-from="16" data-node-to="17"></line>\n<line x1="26.664724349975586" y1="34.68063735961914" x2="33.35383987426758" y2="47.72441864013672" data-type="edge" data-node-from="14" data-node-to="15"></line>\n<line x1="18.303325653076172" y1="37.857967376708984" x2="24.323532104492188" y2="51.73788833618164" data-type="edge" data-node-from="12" data-node-to="13"></line>\n<line x1="7.935193061828613" y1="44.54708480834961" x2="14.624310493469238" y2="56.0858154296875" data-type="edge" data-node-from="10" data-node-to="11"></line>\n<line x1="73.82300567626953" y1="41.871437072753906" x2="87.20124053955078" y2="53.07571029663086" data-type="edge" data-node-from="8" data-node-to="9"></line>\n<line x1="50.243865966796875" y1="36.68737030029297" x2="60.611995697021484" y2="37.857967376708984" data-type="edge" data-node-from="6" data-node-to="7"></line>\n<line x1="42.0496940612793" y1="36.520145416259766" x2="50.243865966796875" y2="36.68737030029297" data-type="edge" data-node-from="5" data-node-to="6"></line>\n<line x1="31.514333724975586" y1="35.85123062133789" x2="42.0496940612793" y2="36.520145416259766" data-type="edge" data-node-from="4" data-node-to="5"></line>\n<line x1="22.65125274658203" y1="37.690738677978516" x2="31.514333724975586" y2="35.85123062133789" data-type="edge" data-node-from="3" data-node-to="4"></line>\n<line x1="15.125994682312012" y1="42.038665771484375" x2="22.65125274658203" y2="37.690738677978516" data-type="edge" data-node-from="2" data-node-to="3"></line>\n<line x1="1.7477586269378662" y1="51.4034309387207" x2="15.125994682312012" y2="42.038665771484375" data-type="edge" data-node-from="1" data-node-to="2"></line>\n<circle r="0.75" cx="1.7477586269378662" cy="51.4034309387207" data-type="element node" data-element-id="1" data-node-id="1" data-label-name="1"></circle>\n<circle r="0.75" cx="15.125994682312012" cy="42.038665771484375" data-type="element node" data-element-id="2" data-node-id="2" data-label-name="2"></circle>\n<circle r="0.75" cx="22.65125274658203" cy="37.690738677978516" data-type="element node" data-element-id="3" data-node-id="3" data-label-name="3"></circle>\n<circle r="0.75" cx="31.514333724975586" cy="35.85123062133789" data-type="element node" data-element-id="4" data-node-id="4" data-label-name="4"></circle>\n<circle r="0.75" cx="42.0496940612793" cy="36.520145416259766" data-type="element node" data-element-id="5" data-node-id="5" data-label-name="5"></circle>\n<circle r="0.75" cx="50.243865966796875" cy="36.68737030029297" data-type="element node" data-element-id="6" data-node-id="6" data-label-name="6"></circle>\n<circle r="0.75" cx="60.611995697021484" cy="37.857967376708984" data-type="element node" data-element-id="7" data-node-id="7" data-label-name="7"></circle>\n<circle r="0.75" cx="73.82300567626953" cy="41.871437072753906" data-type="element node" data-element-id="8" data-node-id="8" data-label-name="8"></circle>\n<circle r="0.75" cx="87.20124053955078" cy="53.07571029663086" data-type="element node" data-element-id="9" data-node-id="9" data-label-name="9"></circle>\n<circle r="0.75" cx="7.935193061828613" cy="44.54708480834961" data-type="element node" data-element-id="10" data-node-id="10" data-label-name="10"></circle>\n<circle r="0.75" cx="14.624310493469238" cy="56.0858154296875" data-type="element node" data-element-id="11" data-node-id="11" data-label-name="11"></circle>\n<circle r="0.75" cx="18.303325653076172" cy="37.857967376708984" data-type="element node" data-element-id="12" data-node-id="12" data-label-name="12"></circle>\n<circle r="0.75" cx="24.323532104492188" cy="51.73788833618164" data-type="element node" data-element-id="13" data-node-id="13" data-label-name="13"></circle>\n<circle r="0.75" cx="26.664724349975586" cy="34.68063735961914" data-type="element node" data-element-id="14" data-node-id="14" data-label-name="14"></circle>\n<circle r="0.75" cx="33.35383987426758" cy="47.72441864013672" data-type="element node" data-element-id="15" data-node-id="15" data-label-name="15"></circle>\n<circle r="0.75" cx="38.537906646728516" cy="32.841129302978516" data-type="element node" data-element-id="16" data-node-id="16" data-label-name="16"></circle>\n<circle r="0.75" cx="40.04296112060547" cy="47.055503845214844" data-type="element node" data-element-id="17" data-node-id="17" data-label-name="17"></circle>\n<circle r="0.75" cx="46.89930725097656" cy="32.841129302978516" data-type="element node" data-element-id="18" data-node-id="18" data-label-name="18"></circle>\n<circle r="0.75" cx="47.066532135009766" cy="48.39332962036133" data-type="element node" data-element-id="19" data-node-id="19" data-label-name="19"></circle>\n<circle r="0.75" cx="56.26407241821289" cy="35.85123062133789" data-type="element node" data-element-id="20" data-node-id="20" data-label-name="20"></circle>\n<circle r="0.75" cx="55.42793273925781" cy="49.229469299316406" data-type="element node" data-element-id="21" data-node-id="21" data-label-name="21"></circle>\n<circle r="0.75" cx="67.97002410888672" cy="35.34954833984375" data-type="element node" data-element-id="22" data-node-id="22" data-label-name="22"></circle>\n<circle r="0.75" cx="66.29774475097656" cy="49.73115158081055" data-type="element node" data-element-id="23" data-node-id="23" data-label-name="23"></circle>'}],
                          'version': '1.0',
                          'data': {'chunk_size': 36, 'image_quality': 70, 'start_frame': 0,
                                   'stop_frame': stop_frame,
                                   'storage_method': 'cache', 'storage': 'local', 'sorting_method': 'lexicographical',
                                   'chunk_type': 'imageset', 'deleted_frames': []}, 'jobs': [
                {'start_frame': 0, 'stop_frame': stop_frame, 'frames': [], 'status': 'completed',
                 'type': 'annotation'}]}
        # Save task.json
        self.create_json_file_by_list(os.path.join(self.img_out, "task.json"), self.task_json)

        # Create annotation.json
        self.annotation_json_path = os.path.join(self.img_out, 'annotations.json')
        self.create_json(self.annotation_json_path)

        # Create and init manifest.json
        self.manifest_json_path = os.path.join(self.data_path, 'manifest.jsonl')
        self.create_json(self.manifest_json_path)
        self.update_json(self.manifest_json_path, "{\"version\":\"1.1\"}")
        self.update_json(self.manifest_json_path, "{\"type\":\"images\"}")
        self.skeleton = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [10, 11], [12, 13], [14, 15],
                         [16, 17], [18, 19], [20, 21], [22, 23]]
        self.out_dir = "D:/vitpose_work_dir/data/global_system/3_pose_estimation/MOCKS/LATERAL"

    @staticmethod
    def create_json(json_file_path):
        with open(json_file_path, 'w'):
            pass

    @staticmethod
    def update_json(json_file_path, new_data):
        with open(json_file_path, 'a') as file:
            file.write(new_data + '\n')

    @staticmethod
    def create_json_file_by_list(json_file_path, data_list):
        with open(json_file_path, 'w') as json_file:
            json.dump(data_list, json_file)

    @staticmethod
    def zip_folder(folder_path, zip_path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_path)
                    zip_file.write(file_path, arcname)

    def execute_self_labeling(self, outputs):
        frame_num = 0
        for batch_item in outputs:
            for i in range(len(batch_item['image_paths'])):
                if self.limit is not None and self.limit == frame_num:
                    break
                image_input_path = batch_item['image_paths'][i]
                image_name = os.path.basename(image_input_path)
                destination_path = os.path.join(self.data_path, image_name)
                shutil.copy(image_input_path, destination_path)
                image_name = image_name[:-4]
                self.update_json(self.manifest_json_path,
                                 "{\"name\":\"" + image_name + "\",\"extension\":\".png\",\"width\":1920,\"height\":1080,\"meta\":{\"related_images\":[]}}")
                keypoints = batch_item['preds'][i, :, :2]
                # if image_name == 'CI521_LR_0_60E7':
                #     image_np = cv2.imread(image_input_path)
                #     set_keypoints_circles(image_np, keypoints, color=(0, 255, 0), args_radius=3)
                #     t = cv2.imwrite(f'{image_name}_2.png', image_np)
                self.save_image(image_input_path, keypoints, self.out_dir)
                point = 0
                for dt in self.annotation_json:
                    for track in dt['tracks']:
                        for element in track['elements']:
                            raw_point = keypoints[point]
                            element['shapes'].append(
                                {'attributes': [], 'frame': frame_num, 'occluded': False, 'outside': False,
                                 'points': [raw_point[0], raw_point[1]], 'rotation': 0.0,
                                 'type': 'points', 'z_order': 0})
                            point = point + 1
                frame_num += 1
        ann = str(self.annotation_json)
        ann = ann.replace(" ", "")
        ann = ann.replace("\'", "\"")
        ann = ann.replace("False", "false")
        self.update_json(self.annotation_json_path, ann)
        lim_info = "" if self.limit is None else "_lim_" + str(self.limit)
        self.zip_folder(self.img_out, os.path.join(self.images_output_dir, self.basename + lim_info + ".zip"))

    def save_image(self, image_path, keypoints, out_dir):
        image_name = os.path.basename(image_path)
        image_np = cv2.imread(image_path)
        out_file = os.path.join(out_dir, f'{image_name[:-4]}_with_pred.jpg')
        ros_text = "Rostrum good"
        view_text = "Lateral view" if "SP" not in image_name else "Dorsal view"
        pd_bbox = get_bbox(keypoints)[0]
        pd_image = self.get_pd_image(image_np.copy(), pd_bbox, keypoints, self.skeleton, ros_text=ros_text,
                                     view_text=view_text)
        cv2.imwrite(out_file, pd_image)

    import json

    @staticmethod
    def leer_json(ruta_archivo):
        """
        Lee un archivo JSON y devuelve su contenido como un diccionario de Python.

        :param ruta_archivo: Ruta del archivo JSON a leer.
        :return: Diccionario con los datos del archivo JSON.
        """
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                datos = json.load(archivo)
            return datos
        except FileNotFoundError:
            print(f"Error: El archivo '{ruta_archivo}' no fue encontrado.")
            return None
        except json.JSONDecodeError:
            print(f"Error: El archivo '{ruta_archivo}' no contiene un JSON válido.")
            return None

    def execute_self_labeling_no_test(self, config, checkpoint, person_results_file=None):
        if person_results_file is not None:
            person_results = self.leer_json(person_results_file)
        out_temporal_file = os.path.join(self.images_output_dir, "test_keypoints.json")
        state_json_data = {"licenses": [{"name": "", "id": 0, "url": ""}],
                           "info": {"contributor": "", "date_created": "", "description": "", "url": "",
                                    "version": "", "year": ""}, "categories": [
                {"id": 1, "name": "skeleton", "supercategory": "",
                 "keypoints": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                               "17", "18", "19", "20", "21", "22", "23"],
                 "skeleton": [[10, 11], [1, 2], [3, 4], [12, 13], [18, 19], [22, 23], [2, 3], [6, 7], [4, 5], [8, 9],
                              [5, 6], [7, 8], [14, 15], [16, 17], [20, 21]]}], "images": [], "annotations": []}

        pose_model = init_pose_model(config, checkpoint, device="cuda:0".lower())
        dataset = pose_model.cfg.data['val']['type']
        dataset_info = pose_model.cfg.data['val'].get('dataset_info', None)
        if dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            dataset_info = DatasetInfo(dataset_info)
        return_heatmap = False
        output_layer_names = None
        assert len(self.img_list) == len(self.dpt_list), "Differences between the images and their depths."
        for i in range(len(self.img_list)):
            image_add = os.path.join(self.image_path, self.img_list[i])
            depth_add = os.path.join(self.depth_path, self.dpt_list[i])
            persres = None
            if person_results_file is not None:
                bbox = self.get_bbox(person_results, image_add)
                persres = [{'bbox': bbox}]

            test = self.img_list[i].replace("CI", "DI")
            test = test.replace("png", "npy")
            assert test == self.dpt_list[i], f"{self.img_list[i]} doesnt have depth image."

            name = self.img_list[i][:-4]

            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                image_add,
                depth_add,
                person_results=persres,
                bbox_thr=None,
                format='xywh',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
            keypoints = pose_results[0]['keypoints'][:, :2]
            bbox, abb = get_bbox(keypoints)
            keypoints_f = flat_keypoints(keypoints)
            add_new_image_to_json_file(image_add, i, state_json_data, keypoints=keypoints_f, bbox=bbox, abb=abb)

            if name == 'CI521_LR_0_60E7':
                image_np = cv2.imread(image_add)
                set_keypoints_circles(image_np, keypoints, color=(0, 255, 0), args_radius=3)
                t = cv2.imwrite(f'{name}_1.png', image_np)

        create_json(out_temporal_file, state_json_data)

    def get_bbox(self, person_results, in_file):
        name = os.path.basename(in_file)
        ide = self.find_dict_by_param(person_results['images'], 'file_name', name)
        image_dic = ide.copy()
        ade = self.find_dict_by_param(person_results['annotations'], 'id', image_dic['id'])
        return np.array(ade['bbox'])

    @staticmethod
    def find_dict_by_param(lis, key, param):
        for dic in lis:
            if dic.get(key) == param:
                return dic
        return None

    def get_pd_image(self, image_np, gt_bbox, pd_keypoint, skeleton, ros_text=None, view_text=None):
        text = f"{view_text}, {ros_text}"
        pd_image = image_np.copy()
        self.set_bbox_rectangle(pd_image, gt_bbox, text, color=(0, 0, 255))
        self.set_keypoint_joint_lines(pd_image, pd_keypoint, skeleton)
        return pd_image

    @staticmethod
    def set_bbox_rectangle(image, bbox, text, color, args_line_width=2):
        x, y, width, height = bbox
        fac = 30
        x = x - fac
        y = y - fac
        cv2.rectangle(image, (round(bbox[0]) - 1 - fac, round(bbox[1]) - 1 - fac),
                      (round(bbox[2]) + round(bbox[0] + fac), round(bbox[3]) + round(bbox[1]) + fac), color,
                      args_line_width)

        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 5)
        text_bg_top_left = (x, y - text_height - 10)  # Arriba del rectángulo
        text_bg_bottom_right = (x + text_width + 10, y)
        text_bg_top_left = (int(text_bg_top_left[0]), int(text_bg_top_left[1]))
        text_bg_bottom_right = (int(text_bg_bottom_right[0]), int(text_bg_bottom_right[1]))

        overlay = image.copy()
        cv2.rectangle(overlay, text_bg_top_left, text_bg_bottom_right, color, -1)
        cv2.addWeighted(overlay, 0.9, image, 1 - 0.9, 0, image)
        text_position = (int(x + 5), int(y - 5))
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

    def set_keypoint_joint_lines(self, image, keypoint, skeleton):
        for union in skeleton:
            point1 = [int(valor) for valor in keypoint[union[0] - 1]]
            point2 = [int(valor) for valor in keypoint[union[1] - 1]]
            image = self.draw_text_on_line(image, point1, point2)

    @staticmethod
    def draw_text_on_line(img, p1, p2, line_color=(0, 255, 255), circle_color=(0, 0, 255), circle_size=10):
        cv2.line(img, p1, p2, line_color, 4)

        cv2.circle(img, (p1[0], p1[1]), circle_size, (0, 0, 0), -1)
        cv2.circle(img, (p1[0], p1[1]), circle_size - 2, circle_color, -1)
        cv2.circle(img, (p2[0], p2[1]), circle_size, (0, 0, 0), -1)
        cv2.circle(img, (p2[0], p2[1]), circle_size - 2, circle_color, -1)

        return img


def remove_first_image_from_cvat_backup(cvat_backup_path):
    # read a zip file
    with zipfile.ZipFile(cvat_backup_path, 'r') as zip_ref:
        zip_ref.extractall(cvat_backup_path[:-4])
    # zip file:
    # - data (directory)
    #   - index.json
    #   - manifest.jsonl
    # - annotations.json
    # - task.json

    # remove the first image from the data directory
    data_dir = os.path.join(cvat_backup_path[:-4], "data")
    images = os.listdir(data_dir)

    name = images[0]
    frame = 0
    delete_image(name, frame, data_dir, cvat_backup_path)

    while True:
        task_json_path = os.path.join(cvat_backup_path[:-4], "task.json")
        with open(task_json_path, 'r') as f:
            task_json = json.load(f)

        # change the deleted frames
        deleted_frames = task_json['data']['deleted_frames']
        if 0 in deleted_frames:
            data_dir = os.path.join(cvat_backup_path[:-4], "data")
            images = os.listdir(data_dir)
            name = images[0]
            delete_image(name, frame, data_dir, cvat_backup_path)
        else:
            break

    # delete index.json file
    index_json_path = os.path.join(data_dir, "index.json")
    if os.path.exists(index_json_path):
        os.remove(index_json_path)

    zip_file_path = cvat_backup_path[:-4] + "_update.zip"
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        # Add the data directory and its contents to the zip file
        for root, dirs, files in os.walk(cvat_backup_path[:-4]):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, cvat_backup_path[:-4])
                zip_ref.write(file_path, arcname)


def delete_image(name, frame, data_dir, cvat_backup_path):
    os.remove(os.path.join(data_dir, name))

    name_man = name[:-4]
    manifest_json_path = os.path.join(data_dir, "manifest.jsonl")
    with open(manifest_json_path, 'r') as f:
        manifest_json = f.readlines()
    # delete the line where is name
    for i in range(len(manifest_json)):
        if name_man in manifest_json[i]:
            del manifest_json[i]
            break
    # manifest_json = manifest_json[0:2] + manifest_json[3:]
    # save the new manifest.jsonl
    with open(manifest_json_path, 'w') as f:
        f.writelines(manifest_json)

    annotations_json_path = os.path.join(cvat_backup_path[:-4], "annotations.json")
    with open(annotations_json_path, 'r') as f:
        annotations_json = json.load(f)

    for element in annotations_json[0]['tracks'][0]['elements']:
        for shape in element['shapes']:
            if shape['frame'] == frame:
                element['shapes'].remove(shape)

    for element in annotations_json[0]['tracks'][0]['elements']:
        for shape in element['shapes']:
            if shape['frame'] > frame:
                shape['frame'] = shape['frame'] - 1
    # save the results of annotations.json
    with open(annotations_json_path, 'w') as f:
        json.dump(annotations_json, f)

    task_json_path = os.path.join(cvat_backup_path[:-4], "task.json")
    with open(task_json_path, 'r') as f:
        task_json = json.load(f)

    # change stop frame
    task_json['data']['stop_frame'] = task_json['data']['stop_frame'] - 1
    task_json['jobs'][0]['stop_frame'] = task_json['jobs'][0]['stop_frame'] - 1
    # change the deleted frames
    task_json['data']['deleted_frames'] = [i - 1 for i in task_json['data']['deleted_frames']]
    if -1 in task_json['data']['deleted_frames']:
        task_json['data']['deleted_frames'].remove(-1)

    # save the results of task.json
    with open(task_json_path, 'w') as f:
        json.dump(task_json, f)


def create_sample_ann_file_to_test(input_path, output_path, outputs=None):
    out_temporal_file = os.path.join(output_path, "test_keypoints.json")
    state_json_data = {"licenses": [{"name": "", "id": 0, "url": ""}],
                       "info": {"contributor": "", "date_created": "", "description": "", "url": "",
                                "version": "", "year": ""}, "categories": [
            {"id": 1, "name": "skeleton", "supercategory": "",
             "keypoints": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                           "17", "18", "19", "20", "21", "22", "23"],
             "skeleton": [[10, 11], [1, 2], [3, 4], [12, 13], [18, 19], [22, 23], [2, 3], [6, 7], [4, 5], [8, 9],
                          [5, 6], [7, 8], [14, 15], [16, 17], [20, 21]]}], "images": [], "annotations": []}
    list_input = os.listdir(input_path)
    if "color_images" in list_input:
        color_images = os.path.join(input_path, "color_images")
    else:
        color_images = input_path
    count = 0
    if outputs is None:
        for image in os.listdir(color_images):
            kp = [1111.93, 136.5, 2, 1110.21, 370.77, 2, 1105.29, 429.73, 2, 1108.9, 497.91, 2, 1109.05,
                  570.84, 2, 1109.44, 627.7, 2, 1105.97, 672.24, 2, 1099.19, 749.69, 2, 1092.74, 821.37,
                  2,
                  1144.9, 339.58, 2, 1065.08, 339.38, 2, 1148.77, 407.67, 2, 1067.11, 409.95, 2, 1148.86,
                  466.24, 2, 1070.16, 467.96, 2, 1144.16, 528.96, 2, 1075.65, 529.31, 2, 1137.01, 597.78,
                  2,
                  1079.88, 598.75, 2, 1130.47, 641.43, 2, 1077.73, 640.58, 2, 1121.46, 696.41, 2,
                  1080.65,
                  694.58, 2]
            bboxp, axp = get_bbox(np.array(
                [[1111.93, 136.5], [1110.21, 370.77], [1105.29, 429.73], [1108.9, 497.91], [1109.05, 570.84],
                 [1109.44, 627.7], [1105.97, 672.24], [1099.19, 749.69], [1092.74, 821.37], [1144.9, 339.58],
                 [1065.08, 339.38], [1148.77, 407.67], [1067.11, 409.95], [1148.86, 466.24], [1070.16, 467.96],
                 [1144.16, 528.96], [1075.65, 529.31], [1137.01, 597.78], [1079.88, 598.75], [1130.47, 641.43],
                 [1077.73, 640.58], [1121.46, 696.41], [1080.65, 694.58]]).reshape(23, 2))
            add_new_image_to_json_file(os.path.join(color_images, image), count, state_json_data, keypoints=kp,
                                       bbox=bboxp, abb=axp)
            count += 1
    else:
        for batch_item in outputs:
            for idx in range(len(batch_item['image_paths'])):
                img_path = batch_item['image_paths'][idx]
                keypoints = batch_item['preds'][idx, :, :2]
                bbox, abb = get_bbox(keypoints)
                keypoints = flat_keypoints(keypoints)

                add_new_image_to_json_file(img_path, count, state_json_data, keypoints=keypoints, bbox=bbox, abb=abb)
                count += 1

    create_json(out_temporal_file, state_json_data)


def get_bbox(keypoints):
    x_min = np.min(keypoints[:, 0])
    y_min = np.min(keypoints[:, 1])
    x_max = np.max(keypoints[:, 0])
    y_max = np.max(keypoints[:, 1])

    width = x_max - x_min
    height = y_max - y_min

    area = width * height

    return [np.float64(x_min), np.float64(y_min), np.float64(width), np.float64(height)], np.float64(area)


def flat_keypoints(keypoints):
    flattened_array = keypoints.flatten()
    result = []

    # Iterar a través del array aplanado y agregar el valor 2 después de cada par
    for i in range(0, len(flattened_array), 2):
        result.append(np.float64(flattened_array[i]))
        result.append(np.float64(flattened_array[i + 1]))
        result.append(2)

    # Convertir la lista a un ndarray
    return result


def create_json(address, data):
    with open(address, 'w') as archivo_json:
        json.dump(data, archivo_json)


def add_new_image_to_json_file(name, id, state_json_data, keypoints=None, bbox=None, abb=None):
    name = os.path.basename(name)
    ide = {"id": id, "width": 1920, "height": 1080, "file_name": name, "license": 0, "flickr_url": "",
           "coco_url": "", "date_captured": 0}
    image_dic = ide.copy()

    ade = {"id": id, "image_id": id, "category_id": 1, "segmentation": [], "area": abb,
           "bbox": bbox, "iscrowd": 0,
           "attributes": {"occluded": False, "track_id": 0, "keyframe": True},
           "keypoints": keypoints, "num_keypoints": 23}

    ann_dic = ade.copy()
    state_json_data['images'].append(image_dic)
    state_json_data['annotations'].append(ann_dic)
