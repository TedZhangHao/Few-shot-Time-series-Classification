# Few-shot Time-Series Classification (FsTSC)
FsTSC is an open-source deep learning platform for Few-shot time-series classification, aiming at providing efficient and accurate Few-shot solution.

We provide user-friendly code base for evaluating deep learning models that delving TSC problems. 

We also provide a unique data augmentation approach by utilizing short-time Fourier transform (STFT) and random erasure: 
1. Transform time-series data to spectrogram image using STFT.
2. Apply random erasure on spectrogram image for data augmentation (generation).
3. Align spectrogram and time-series data, which are reckoned as 2 different modalities, to construct a multi-modal dataset.

A novel feature-level multi-modal networks, **Sequence-Spectrogram Fusion Network (SSFN)**, is developed for fitting our data augmentation approach.

Our few-shot framework is demonstrated efficient as well as effective for wind turbine health diagnosis.
