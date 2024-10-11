# Few-shot Time-Series Classification (FsTSC)
FsTSC is an open-source deep learning platform for Few-shot time-series classification, aiming at providing efficient and accurate Few-shot solution.


We provide user-friendly code base for evaluating deep learning models that delving TSC problems. 

We also provide a unique data augmentation approach by utilizing short-time Fourier transform (STFT) and random erasure: 
1. Transform time-series data to spectrogram image using STFT.
2. Apply random erasure on spectrogram image for data augmentation (generation).
3. Align spectrogram and time-series data, which are reckoned as 2 different modalities, to construct a multi-modal dataset.

A novel feature-level multi-modal networks, **Sequence-Spectrogram Fusion Network (SSFN)**, is developed for fitting our data augmentation approach.

Our few-shot framework is demonstrated efficient as well as effective for wind turbine health diagnosis. 
## Our WorkfLow
The workflow is depicted as follows:
<p align="center">
<img src=".\pro_pic\Workflow.png" height = "600", width = "1000", alt="" align=center />
</p>

## Usage

1. Install Python 3.9. For install required packages, execute the following command.

```
pip install -r requirements.txt
```
2. The source dataset is currently reserved, yet we released the processed dataset under the folder `.\data_provider\dataset\Example_WTIL` for directly uasge and validation. 
<p align="center">
<img src=".\pro_pic\Impact_trail.png" height = "400", width = "700", alt="" align=center />
</p>

3.For training and evaluating our proposed model SSFN immediately, you can:
```
bash ./bash_sh/SSFN.sh
```
or run the `./run.py` with further parameter tuning and other benchmark models.

## Contact
If you have any questions or suggestions, feel free to contact:

- Hao Zhang (haozhang1639@163.com)

## Acknowledgement
we utilized the models form tsai(https://github.com/timeseriesAI/tsai) and Time-series-Library(https://github.com/thuml/Time-Series-Library) as comparsions, 
