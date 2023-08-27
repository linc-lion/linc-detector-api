import unittest
import os
import json
from app import app
from werkzeug.datastructures import FileStorage
from unittest.mock import patch


class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_annotate_image(self):
        with open('static/test_image.jpg', 'rb') as image_file:
            data = {
                'file': (image_file, 'test_image.jpg')
            }

            response = self.app.post('/v1/annotate', data=data, content_type='multipart/form-data')
            json_data = json.loads(response.data.decode('utf-8'))

            self.assertEqual(response.status_code, 200)
            self.assertIn('input_image', json_data)
            self.assertIn('annotated_image', json_data)
            self.assertIn('bounding_box_coords', json_data)

    def test_annotate_image_no_file_exception(self):
        response = self.app.post('/v1/annotate')
        json_data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(response.status_code, 500)
        self.assertIn('error', json_data)
        self.assertTrue(json_data['error'].startswith('No file sent'))


if __name__ == '__main__':
    unittest.main()
