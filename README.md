# colour_project
## Table of contents
- [Theoretical justification](#color-theory)
- [Our project](#our-project)
- [Color region recognition](#сolor-region-recognition)
- [Linear colour mapping](#color-correction-methods-and-deltae-metric)
- [Histogram matching](#histogram-matching-method)
---
## Color theory

**Light..**

Light is a form of electromagnetic radiation with both objective characteristics, such as spectral power distribution (SPD), and subjective characteristics, such as color, which is determined by the resulting visual sensation in the human brain.

Color can be described using various models, known as color spaces, where each color is encoded by specific coordinates. The most popular color spaces include RGB, which is commonly used for scanners and digital displays, and XYZ, a color space closely approximating real cone activations in the human eye and often used as a standard for evaluating light perception.

By applying transfer functions, you can convert between different color spaces and describe light with a specific SPD in any chosen color space.

**.. and its reproduction**

Imagine sunlight characterized by a particular spectral power distribution (SPD) falling on a surface, such as a blue one in this example. The light is then reflected, and its SPD changes. When this reflected light reaches the human eye, the cones in the retina are activated, leading to a specific color perception. This perception can be described by the X, Y, and Z values in the XYZ color space, which roughly correspond to the activation of the red, green, and blue cones, respectively.

We can also imitate this phenomena. To do so we detect the reflected light using scanners that measure how much light passes through red, green, and blue filters, described as R, G, and B in the RGB color space. These values can then be used to reproduce the physical phenomenon using red, green, and blue lamps. The light produced by these lamps may have a different SPD from the original, but in an ideal case, it should perfectly imitate the visual sensation perceived by humans.

However, in real life, achieving perfect imitation is challenging due to scanner inaccuracies. Manufacturing identical scanners with identical filters and other components is complex, and as a result, viewing RGB images from different scanners on the same monitor can yield different results.
[<img src="/images/Color1.png">]()

## Our project

**Aim**

The aim of our project is to create a color correction pipeline that brings the values obtained from scanners as close as possible to the ground truth values — the spectral power distribution (SPD) measured by a spectrophotometer. For compatibility, by default we work both with the spectral power distribution and RGB values encoded in the XYZ color space.

**Project pipeline**

First, we assess the XYZ values of landmark objects on a calibration palette using both the spectrophotometer (ground truth) and the scanner under investigation. These values are used for color correction and initial difference (deltaE) evaluation. Several color correction methods were implemented in our project, including PLS, Lasso, Vote, and TPS. Initially, we deployed these methods in the XYZ color space and then experimented in the RGB color space. The differences after correction were evaluated to determine the best color correction method.
[<img src="/images/Color2.png">]()

## Color region recognition
Identifying the scanner values for the landmark objects was the first task. To achieve this, we developed an object detection pipeline. The input images were stored as zarr files. To minimize computational load, we initially worked with a small layer of the zarr: aligned, flipped, and rotated the image as needed and detected objects of interest. Using the coordinates obtained from the small layer, we then returned to the original coordinates to calculate the RGB values of each object using the full-resolution layer. The object detection pipeline was implemented using the OpenCV library and demonstrated reproducible results on images from different scanners.
[<img src="/images/Color3.png">]()
[<img src="/images/Color4.png">]()

## Color correction methods and deltaE metric. 

In this section, we delve into the color correction methods applied in our project. After identifying the coordinates of the color palette circles, we proceeded to train several regression models on this test data. Specifically, we fitted four different regressors: Partial Least Squares (PLS), Lasso, Thin Plate Spline, and a Voting Regressor, which combines the predictions of the other models.

The fitting process was based on the training data of colors within the selected color space, compared to reference data obtained by converting spectral measurements. Prior to training, it was essential to calculate the DeltaE metric, which quantifies the perceived color difference.

After the models were trained, we used them to predict the corrected colors on the image. By extracting the corrected colors using the identified coordinates, we could then compute the final DeltaE metric, which serves as a measure of the accuracy of our color correction methods.
[<img src="/images/Color5.png">]()
[<img src="/images/Color6.png">]()
### Color correction results. deltaE (by object)
[<img src="/images/Color7.png">]()
### Color correction results. XYZ values (by object)
[<img src="/images/Color8.png">]()
[<img src="/images/Color9.png">]()
## Histogram matching method
[<img src="/images/Color10.png">]()
[<img src="/images/Color11.png">]()
[<img src="/images/Color12.png">]()
### Histogram matching results
[<img src="/images/Color13.png">]()