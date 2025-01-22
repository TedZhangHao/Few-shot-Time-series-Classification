# Few-shot Time-Series Classification (FsTSC)
FsTSC is an open-source deep learning platform for Few-shot time-series classification, aiming at providing efficient and accurate Few-shot solution.


We provide a user-friendly code base for evaluating off-the-shelf models that focus on TSC problems. 

We also provide a unique data augmentation approach by utilizing short-time Fourier transform (STFT) and random erasure: 
1. Transform time-series data to spectrogram images using STFT.
2. Apply random erasure [[AAAI 2020]](https://aaai.org/papers/13001-random-erasing-data-augmentation/) on spectrogram image for data augmentation (generation).
3. Align spectrogram and time-series data, which are reckoned as 2 different modalities, to construct a multi-modal dataset.

A novel feature-level multi-modal networks, **Sequence-Spectrogram Fusion Network (SSFN)**, is developed for fitting our data augmentation approach.

Our few-shot framework is demonstrated efficient as well as effective on the task of wind turbine health diagnosis. All the details of our approach are introduced in our published paper in Advanced Engineering Informatics (AEI) *Sequence-Spectrogram Fusion Network for Machinery Diagnosis through Few-Shot Time-Series Classification* https://www.sciencedirect.com/science/article/pii/S147403462400627X. 
## Our Workflow
The workflow is depicted as follows:
<p align="center">
<img src=".\pro_pic\Workflow.png" height = "500", width = "900", alt="" align=center />
</p>

## Examples
Processing data from time-series -> spectrogram -> augmented spectrogram (random erasure)
<p align="center">
<img src=".\pro_pic\WTIL_TS.png" height = "180", width = "250"/>
<img src=".\pro_pic\WTIL_STFT.png" height = "180", width = "280"/>
<img src=".\pro_pic\WTIL_STFT_Aug.png" height = "180", width = "280"/>
</p>

## Usage

1. Install Python 3.9. To install the required packages, execute the following command.

```
pip install -r requirements.txt
```
2. The source dataset is currently restricted; however, we have released the processed dataset under the folder `./data_provider/dataset/Example_WTIL` for direct usage and validation. 
<p align="center">
<img src=".\pro_pic\Impact_trail.png" height = "400", width = "500", alt="" align=center />
</p>

3. For training and evaluating our proposed model SSFN immediately, you can:
```
bash ./bash_sh/SSFN.sh
```
or run the `./run.py` with further parameter tuning and other benchmark models.

4. For data processing, STFT, data augmentation, and modality aligning, we provide an example processing file under the folder `./data_provider/`.
   1. For preprocess time-series data and dataset split: `./data_provider/Preprocess_Dataset.py`.
   2. For spectrogram generation transformed from time-series data: `./data_provider/Spectrogram_Generation.py`.
   3. For data augmentation: `./data_provider/Augmentation.py`.
   **Note**: Since we have not uploaded the source data, the Python files above cannot be run directly.  

## Contact
If you have any questions or suggestions, feel free to contact:

- Hao Zhang (haozhang1639@163.com)

## Acknowledgement
We employed the benchmark models (code, including layers and utils) from tsai (https://github.com/timeseriesAI/tsai) and Time-series-Library (https://github.com/thuml/Time-Series-Library) for convenient usage.

The experiment dataset is provided by Shandong University Key Laboratory of Machine Intelligence & System Control, Ministry of Education.

I am especially grateful to my advisor Teng Li, who proffers tremendous support to this work.
