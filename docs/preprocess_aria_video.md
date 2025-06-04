
# Preprocessing Explained

The raw data preprocessing script will do the following work.

Extract the raw data with metadata from the VRS file and MPS metadata in stage one:
* Read the correct (online-calibrated) camera intrinsics.
* Estimate the correct timestamp for each camera stream, the extrinsics and transform that into the camera to world transform matrix.
* For a rolling shutter (RGB) camera, we will generate the transformation matrix at the start, center, and end time respectively. This could be the sufficient information needed to compensate the rolling shutter camera in downstream applications.
* Acquire corresponding device linear&angular velocity according to the timestamp. This has not been tested in downstream app though.
* Read the exposure time and analogy gain for each frame.

Rectify all the RGB camera streams with
* Rectified lens-shading model
* Rectified rolling-shutter row index model
* Reprojected sparse depth from the semi-dense point map

Running the above script will automatically generate a nerfstudio like transform json file for each of the raw camera stream:
```
$DATA_PROCESSED_DIR
- camera-rgb-images
- camera-rgb-transforms.json
- camera-slam-left-images
- camera-slam-left-transforms.json
- camera-slam-right-images
- camera-slam-right-transforms.json
```


## File format of the preprocessed scripts

Running the script will automatically generate a nerfstudio like transform json file for each of the raw camera stream, which in the following data structure in the output folder:
```
$DATA_PROCESSED_DIR
- camera-rgb-images
- camera-rgb-transforms.json
- camera-slam-left-images
- camera-slam-left-transforms.json
- camera-slam-right-images
- camera-slam-right-transforms.json
```
where 'camera-rgb-images', 'camera-slam-left-images', 'camera-slam-right-images' are the raw images folders for each, and each transform json '*-transforms.json' file, it encodes the meta data for **each camera raw data** as following:
``` json
    "camera_model": "FISHEYE624",
    "frames": [
        {
            "fl_x": 1216.7952880859375,
            "fl_y": 1216.7952880859375,
            "cx": 1459.875,
            "cy": 1441.5284423828125,
            # The fisheye distortion parameters
            "distortion_params": [...],
            "w": 2880,
            "h": 2880,
            "file_path": "camera-rgb-images/camera-rgb_9188421039900.0.png",
            "camera_modality": "rgb",
            # 4x4 matrix camera to world. For rolling-shutter (RGB) camera, this is the transformation corresponds to the center row
            "transform_matrix": [[...]],
            # 4x4 matrix camera to world. This is the first row pose of a rolling shutter camera.
            "transform_matrix_read_start": [[...]],
            # 4x4 matrix camera to world. This is the last row pose of a rolling shutter camera.
            "transform_matrix_read_end": [[...]],
            # 4x4 matrix camera to world. This is the pose of the center-row rolling shutter camera after the full exposure time.
            "transform_matrix_read_center_exposure_end": [[...]],
            # The linear velocity in the device frame
            "device_linear_velocity": [...],
            # The angular velocity in the device frame
            "device_angular_velocity": [...],
            # The Aria sensor timestamp (in nanosecond)
            "timestamp": 9188421039900.0,
            # the exposure time in seconds
            "exposure_duration_s": 0.001464,
            # the sensor analog gain
            "gain": 3.002932548522949,
            # camera-rgb for the RGB camera, camera-slam-left or camera-slam-right for the SLAM cameras.
            "camera_name": "camera-rgb"
        }
        # all the rest frame list in the same format
        # ......
        # ......
    ],
    "camera_label": "camera-rgb",
    # The 4x4 transform matrix (camera2world) of device CPF coordinate.
    "transform_cpf": [[]],
```

## Use factory calibration instead of online calibration
In the script, there is an option to used factory calibration instead of online calibrated camera. We don't recommend this in the downstream reconstruction since we have concluded the online device calibration is crucial in particular using RGB camera as input.

Note the improvement over factory calibration will only be valid when the visual inertial bundle adjustment (VIBA) is enabled when running MPS. Without VIBA, there will not be much performance gain using online calibration.

To generate factory calibrated data, run with the mode "--use_factory_calib" in the [preprocessing script](#aria-preprocess-script). It will generate all the results with a post-fix called "-factory-calib" as following:
```
$DATA_PROCESSED_DIR
- camera-rgb-images-factory-calib
- camera-rgb-transforms-factory-calib.json
- camera-slam-left-images-factory-calib
- camera-slam-left-transforms-factory-calib.json
- camera-slam-right-images-factory-calib
- camera-slam-right-transforms-factory-calib.json
```

## Generate rectified images and sparse visible point cloud

In the second stage of the preprocessing, according to the provided the pinhole camera model, the script will further generate the rectified images for each stream. The pinhole camera model parameters were set in the [preprocessing script](#aria-preprocess-script) with the following config
``` bash
--rectified_rgb_focal 1200 \
--rectified_rgb_size 2000 \
--rectified_monochrome_focal 180 \
--rectified_monochrome_height 480 \
```
which set the focal and image size for rectified image streams. Note for RGB camera, there are two image resolution options, the above focal length and image size are for the full image resolution size (2880x2880). For half resolution RGB image (1408x1408), a reasonable estimate to retain the original FoV and size of the image can be
``` bash
--rectified_rgb_focal 600 \
--rectified_rgb_size 1000 \
```

It will generate the following rectified camera stream output with an explanation of each file in the comments
``` bash
- camera-rgb-rectified-1200-h1600
----images                      # The rectified RGB images
----transforms.json             # The transform json file
----vignette.png                # A rectified lens shading model for RGB image
----mask.png                    # A rectified binary mask region for the valid pixels in the lens shading model for RGB image.
----semidense_points.csv.gz     # a symbolic link to the input Aria MPS semi-dense points
- camera-slam-left-rectified-180-h480
----images                      # The rectified SLAM images
----sparse_depth                # A sparse set of depth calculated from the visible semi-dense points in each frame. Only applicable to SLAM images.
----transforms.json             # The transform json file
----vignette.png                # A rectified lens shading model for the SLAM monochrome image
----semidense_points.csv.gz     # a symbolic link to the input Aria MPS semi-dense points
- camera-slam-right-rectified-180-h480
```

## The rectification script will do the following work under the hood:

* Rectify the raw image as well as the vignette image for RGB and SLAM cameras. Note the two cameras models have different lens shading model. The RGB camera recorded at half resolution (1408x1408) is not simply a resize of full resolution image. The rectification will take care of this as well.
* For RGB image, there is a separate rectified mask image indicating the valid pixels regions while SLAM cameras do not.
* For SLAM cameras, we also compute the a sparse depth map, given the semi-dense point cloud and semi-dense observations. It is based on reprojecting the visible tracked points in each slam view.

## Skip preprocessing a camera modality

When setting the rectified_*_focal number smaller than 0, it will skip preprocessing the target modality. For example, when chose
```
--rectified_rgb_focal -1
```
It will skip generate rectified image data for RGB stream.
