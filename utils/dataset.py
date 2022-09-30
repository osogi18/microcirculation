import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import json


class EyeDataset(Dataset):
    """
    Класс датасета, организующий загрузку и получение изображений и соответствующих разметок
    """

    def __init__(self, paths: str, transform = None):
        self.class_ids = {"vessel": 1}

        self.paths = paths
        self.transform = transform

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image / 255, dtype=np.float32)
        return image

    @staticmethod
    def parse_polygon(coordinates, image_size):
        mask = np.zeros(image_size, dtype=np.float32)

        if len(coordinates) == 1:
            points = [np.int32(coordinates)]
            cv2.fillPoly(mask, points, 1)
        else:
            points = [np.int32([coordinates[0]])]
            cv2.fillPoly(mask, points, 1)

            for polygon in coordinates[1:]:
                points = [np.int32([polygon])]
                cv2.fillPoly(mask, points, 0)

        return mask

    @staticmethod
    def parse_mask(shape: dict, image_size: tuple) -> np.ndarray:
        """
        Метод для парсинга фигур из geojson файла
        """
        mask = np.zeros(image_size, dtype=np.float32)
        coordinates = shape['coordinates']
        if shape['type'] == 'MultiPolygon':
            for polygon in coordinates:
                mask += EyeDataset.parse_polygon(polygon, image_size)
        else:
            mask += EyeDataset.parse_polygon(coordinates, image_size)

        return mask

    def read_layout(self, path: str, image_size: tuple) -> np.ndarray:
        """
        Метод для чтения geojson разметки и перевода в numpy маску
        """
        with open(path, 'r', encoding='cp1251') as f:  # some files contain cyrillic letters, thus cp1251
            json_contents = json.load(f)

        num_channels = 1 + max(self.class_ids.values())
        mask_channels = [np.zeros(image_size, dtype=np.float32) for _ in range(num_channels)]
        mask = np.zeros(image_size, dtype=np.float32)

        if type(json_contents) == dict and json_contents['type'] == 'FeatureCollection':
            features = json_contents['features']
        elif type(json_contents) == list:
            features = json_contents
        else:
            features = [json_contents]

        for shape in features:
            channel_id = self.class_ids["vessel"]
            mask = self.parse_mask(shape['geometry'], image_size)
            mask_channels[channel_id] = np.maximum(mask_channels[channel_id], mask)

        mask_channels[0] = 1 - np.max(mask_channels[1:], axis=0)

        return mask_channels[0]

    def __getitem__(self, idx: int) -> dict:
        # Достаём имя файла по индексу
        image_path = self.paths[idx]

        json_path = image_path.replace("png", "geojson")

        image = self.read_image(image_path)

        mask = self.read_layout(json_path, image.shape[:2])

        sample = {'image': image,
                  'mask': mask}

        if self.transform is not None:
            sample = self.transform(**sample)

        image = sample['image']
        mask = sample['mask']

        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(np.array(image, dtype=np.float))
        image = image.type(torch.FloatTensor)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.from_numpy(np.array(mask == 0, dtype=np.uint8))
        return image, mask

    def __len__(self):
        return len(self.paths)

    # Метод для проверки состояния датасета
    def make_report(self):
      reports = []
      if (not self.data_folder):
        reports.append("Путь к датасету не указан")
      if (len(self._image_files) == 0):
        reports.append("Изображения для распознавания не найдены")
      else:
        reports.append(f"Найдено {len(self._image_files)} изображений")
      cnt_images_without_masks = sum([1 - len(glob.glob(filepath.replace("png", "geojson"))) for filepath in self._image_files])
      if cnt_images_without_masks > 0:
        reports.append(f"Найдено {cnt_images_without_masks} изображений без разметки")
      else:
        reports.append(f"Для всех изображений есть файл разметки")
      return reports


class PatchEyeDataset(Dataset):
    """
    Класс датасета, организующий загрузку и получение изображений и соответствующих разметок
    """

    def __init__(self, paths: str, transform = None):
        self.class_ids = {"vessel": 1}

        self.paths = paths
        self.transform = transform

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image / 255, dtype=np.float32)
        return image

    def __getitem__(self, idx: int) -> dict:
        image_path = self.paths[idx]
        mask_path = image_path.split('.')[0] + '_mask.jpg'

        image = self.read_image(image_path)
        mask = self.read_image(mask_path)

        sample = {'image': image,
                  'mask': mask}

        if self.transform is not None:
            sample = self.transform(**sample)

        image = sample['image']
        mask = sample['mask']

        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(np.array(image, dtype=np.float))
        image = image.type(torch.FloatTensor)
        mask = np.expand_dims(mask[:, :, 0], axis=-1)
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        return image, mask

    def __len__(self):
        return len(self.paths)


class PSEyeDataset(Dataset):
    """
    Класс датасета, организующий загрузку и получение изображений и соответствующих разметок
    """

    def __init__(self, paths: str, transform = None):
        self.class_ids = {"vessel": 1}

        self.paths = paths
        self.transform = transform

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image / 255, dtype=np.float32)
        return image

    def __getitem__(self, idx: int) -> dict:
        image_path = self.paths[idx]
        mask_path = image_path.replace('images', 'masks')

        image = self.read_image(image_path)
        mask = self.read_image(mask_path)

        sample = {'image': image,
                  'mask': mask}

        if self.transform is not None:
            sample = self.transform(**sample)

        image = sample['image']
        mask = sample['mask']

        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(np.array(image, dtype=np.float))
        image = image.type(torch.FloatTensor)
        mask = np.expand_dims(mask[:, :, 0], axis=-1)
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        return image, mask

    def __len__(self):
        return len(self.paths)