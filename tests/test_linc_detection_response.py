import unittest
from domain.linc_detection_response import LincDetectionResponse


class TestLincDetectionResponse(unittest.TestCase):
    def test_successful_response(self):
        # Create a successful response instance
        bounding_box_coords = {"sample": [0, 0, 100, 100]}
        response = LincDetectionResponse(bounding_box_coords=bounding_box_coords)

        # Assert that the response is an instance of LincDetectionResponse
        self.assertIsInstance(response, LincDetectionResponse)

        # Assert that bounding_box_coords is present and not None
        self.assertIsNotNone(response.bounding_box_coords)

        self.assertIn("sample", response.bounding_box_coords)

        # Assert that error_message is None
        self.assertIsNone(response.error_message)

    def test_failure_response(self):
        # Create a failure response instance
        error_message = "No lion detected in image"
        response = LincDetectionResponse(error_message=error_message)

        # Assert that the response is an instance of LincDetectionResponse
        self.assertIsInstance(response, LincDetectionResponse)

        # Assert that bounding_box_coords is None
        self.assertIsNone(response.bounding_box_coords)

        # Assert that error_message is not None and has the expected error message
        self.assertIsNotNone(response.error_message)
        self.assertEqual(response.error_message, error_message)


if __name__ == '__main__':
    unittest.main()
