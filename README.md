# Final Project: Advanced Machine Learning Theory at Sungkyunkwan University   
   
### Overview   
This repository contains the final project for the Advanced Machine Learning Theory course under the Department of Data Science, Sungkyunkwan University.    
The project implements and extends the findings from the paper "Are Transformers Effective for Time Series Forecasting?".   
Link: https://arxiv.org/pdf/2205.13504   
Github(Official): https://github.com/cure-lab/LTSF-Linear  
ETTh1 Dataset:  
    - AutoFormer Github(Official): https://github.com/thuml/Autoformer  
    - https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy  
   
### Objectives   
1. **Paper Implementation**: Implement the models and methods presented in the paper "Are Transformers Effective for Time Series Forecasting?".   
2. **Model Improvement**: Develop and improve upon the models suggested in the paper, with a focus on enhancing performance for long-term time series forecasting tasks.   
3. **Model Combination**: Specifically, combine CNN and NLinear models to improve forecasting accuracy for univariate time series data.   
   
### Data and Task
The dataset used in this project is tailored for long-term time series forecasting. The focus is on univariate time series, aiming to predict future values based on past observations.   
   
### Approach   
- **Initial Implementation**: Recreate the models from the paper to establish a baseline.   
- **Model Enhancement**: Introduce improvements to the baseline models by integrating CNN and NLinear architectures.   
- **Evaluation**: Assess the performance of the models using metrics such as MAE, RMSE on both validation and test sets.   
- **Model Combination**:    
  - **NDLinear Model**: Combine NLinear and DLinear models to enhance forecasting accuracy.   
  - **CNN + NLinear Model**: Apply a combination of CNN and NLinear models for improved performance.   
  - **Stacking with Attention Mechanism**: Implement a stacking structure utilizing attention mechanisms to combine all trained time series forecasting models.   
- **Custom Loss**:   
  - Develop and apply custom loss functions tailored to the specific characteristics of long-term time series data.   
  - Specifically, the custom loss function increases the weight of errors as the time step moves further from the encoder, addressing the challenge of predicting distant future values.   
  - This custom loss was applied to all models for comparison.   
   
### Repository Structure
ltsf_linear/  
├── common/  
│   ├── __init__.py  
│   ├── exception.py  
│   ├── logger.py  
│   ├── metrics.py  
│   └── visualize.py  
├── config/  
│   ├── __init__.py  
│   └── config.py  
├── data/  
│   ├── __init__.py  
│   ├── data_factor.py  
│   ├── data_loader.py  
│   └── data_loader_old.py  
├── dataset/  
├── main.py  
├── model/  
│   ├── __init__.py  
│   ├── cnn_nlinear/  
│   │   ├── __init__.py  
│   │   └── execute_module.py  
│   ├── hybrid/  
│   │   ├── __init__.py  
│   │   └── execute_module.py  
│   ├── hyperoptimize.py  
│   └── nlinear/  
│       ├── __init__.py  
│       └── execute_module.py  
└── utils/  
    ├── __init__.py  
    ├── metrics.py  
    ├── timefeatures.py  
    └── tools.py    
   
### Getting Started
1. Clone the repository:
    ```bash
    git clone https://github.com/YongKyeom/ltsf_linear.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ltsf_linear
    ```
3. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the model training script:
    ```bash
    python main.py
    ```
5. Run inference result and calculate test metrics:
    ```bash
    inference_result.ipynb
    ```


### Contributions
Contributions to improve this project are welcome. Please feel free to submit pull requests or open issues for discussion.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
