import json
import os
import re
from itertools import chain


class IAWJSONDataset:
    def __init__(
        self,
        dataset_path: str = "data",
        dataset_json_name: str = r"IkeaAssemblyInstructionDatasetFinal.json",
    ):
        self.dataset_path = dataset_path
        self.dataset_json_name = dataset_json_name
        self.dataset_json_pathname = os.path.join(dataset_path, dataset_json_name)
        with open(self.dataset_json_pathname) as json_file:
            self._dataset = json.load(json_file)

    def get_video_list(self):
        return list(chain.from_iterable([d["videoList"] for d in self._dataset]))

    def get_video_annotation_list(self):
        return list(chain.from_iterable([v["annotation"] for v in self.get_video_list()]))

    @staticmethod
    def video_url_to_id(video_url):
        return video_url.split("https://www.youtube.com/watch?v=")[-1]

    def get_video_pathname_list(self, name_template: str = r"{}.mp4"):
        ret = []
        for furniture in self._dataset:
            for video in furniture["videoList"]:
                video_id = self.video_url_to_id(video["url"])
                ret.append(
                    os.path.join(
                        self.dataset_path,
                        "Furniture",
                        furniture["subCategory"],
                        furniture["id"],
                        "video",
                        name_template.format(video_id),
                    )
                )
        return ret

    def get_step_diagram_pathname_list(
        self, sub_category: str, furniture_id: str, name_template: str = r"step-(\d+).png"
    ):
        reg = re.compile(name_template)
        diagram_path = os.path.join(self.dataset_path, "Furniture", sub_category, furniture_id, "step")
        ret = []
        for image in os.listdir(diagram_path):
            if a := reg.match(image):
                ret.append((int(a.group(1)), image))
        ret.sort(key=lambda x: x[0])
        return [os.path.join(diagram_path, image) for _, image in ret]

    def get_video_pathname(self, sub_category: str, furniture_id: str, video_id: str, name_template: str = r"{}.mp4"):
        return os.path.join(
            self.dataset_path,
            "Furniture",
            sub_category,
            furniture_id,
            "video",
            name_template.format(video_id),
        )

    def get_dataset(self):
        return self._dataset
