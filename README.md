# 데이터 획득


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgbm
from sklearn.preprocessing import MinMaxScaler
```


```python
# 데이터 경로
data_path = '/content/drive/MyDrive/cloud_ai/프로젝트2/'

train = pd.read_csv( data_path + 'train.csv', index_col='id')
test  = pd.read_csv( data_path + 'test.csv', index_col='id')
submission = pd.read_csv( data_path + 'sample_submission.csv')

# original 데이터 불러오기
orig_train = pd.read_csv('/content/drive/MyDrive/cloud_ai/프로젝트2/creditcard.csv')
```

- orinal 데이터까지 학습하면 데이터 양이 늘어났기 때문에 학습 효과가 상승할 것이다
- 하지만 kaggle에서 orinial 데이터는 주어진 데이터와 특성이 모두 같지는 않다고 한다 -> 주어진 데이터와 비교하여 사용할지 결정


```python
train.head(2)
```





  <div id="df-fb8851af-3d78-4596-876c-957c4c32e013">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.074329</td>
      <td>-0.129425</td>
      <td>-1.137418</td>
      <td>0.412846</td>
      <td>-0.192638</td>
      <td>-1.210144</td>
      <td>0.110697</td>
      <td>-0.263477</td>
      <td>0.742144</td>
      <td>...</td>
      <td>-0.334701</td>
      <td>-0.887840</td>
      <td>0.336701</td>
      <td>-0.110835</td>
      <td>-0.291459</td>
      <td>0.207733</td>
      <td>-0.076576</td>
      <td>-0.059577</td>
      <td>1.98</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.998827</td>
      <td>-1.250891</td>
      <td>-0.520969</td>
      <td>-0.894539</td>
      <td>-1.122528</td>
      <td>-0.270866</td>
      <td>-1.029289</td>
      <td>0.050198</td>
      <td>-0.109948</td>
      <td>...</td>
      <td>0.054848</td>
      <td>-0.038367</td>
      <td>0.133518</td>
      <td>-0.461928</td>
      <td>-0.465491</td>
      <td>-0.464655</td>
      <td>-0.009413</td>
      <td>-0.038238</td>
      <td>84.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fb8851af-3d78-4596-876c-957c4c32e013')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-fb8851af-3d78-4596-876c-957c4c32e013 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fb8851af-3d78-4596-876c-957c4c32e013');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
test.head(2)
```





  <div id="df-7bdd0273-5613-443a-abe9-8baf85c7f7d7">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>219129</th>
      <td>120580.0</td>
      <td>2.115519</td>
      <td>-0.691809</td>
      <td>-1.305514</td>
      <td>-0.685655</td>
      <td>-0.641265</td>
      <td>-0.764784</td>
      <td>-0.924262</td>
      <td>-0.023030</td>
      <td>-0.230126</td>
      <td>...</td>
      <td>0.067367</td>
      <td>0.241708</td>
      <td>0.682524</td>
      <td>0.037769</td>
      <td>-0.546859</td>
      <td>-0.123055</td>
      <td>-0.084889</td>
      <td>0.004720</td>
      <td>-0.021944</td>
      <td>29.95</td>
    </tr>
    <tr>
      <th>219130</th>
      <td>120580.0</td>
      <td>1.743525</td>
      <td>-1.681429</td>
      <td>-0.547387</td>
      <td>-1.061113</td>
      <td>-0.695825</td>
      <td>2.458824</td>
      <td>-1.632859</td>
      <td>1.073529</td>
      <td>1.068183</td>
      <td>...</td>
      <td>0.441788</td>
      <td>0.543278</td>
      <td>1.294571</td>
      <td>0.309541</td>
      <td>3.703925</td>
      <td>-0.242579</td>
      <td>0.068708</td>
      <td>0.002629</td>
      <td>0.064690</td>
      <td>163.50</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 30 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7bdd0273-5613-443a-abe9-8baf85c7f7d7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7bdd0273-5613-443a-abe9-8baf85c7f7d7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7bdd0273-5613-443a-abe9-8baf85c7f7d7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
orig_train.head(2)
```





  <div id="df-d992d11d-0339-4b3f-863f-dbc7e74f32b0">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d992d11d-0339-4b3f-863f-dbc7e74f32b0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d992d11d-0339-4b3f-863f-dbc7e74f32b0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d992d11d-0339-4b3f-863f-dbc7e74f32b0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
train.shape, test.shape, submission.shape, orig_train.shape
```




    ((219129, 31), (146087, 30), (146087, 2), (284807, 31))



# 데이터 준비 + EDA

## 피처 요약 함수 구성 - 차후 계속 사용(업그레이드)


```python
def summary_feature_info( df, train=train ):
    sum_df = pd.DataFrame( train.dtypes, columns=['type'])   
    # 인덱스 -> 컬럼으로 이동
    sum_df = sum_df.reset_index()
    # 컬럼명 index -> feature_name 변경
    sum_df.rename( columns={ 'index':'feature_name' }, inplace=True)

    # 결측치수
    sum_df['결측치수'] = train.isnull().sum().values

    # 고유값수
    sum_df['고유값수'] = train.nunique().values

    # 샘플값0~2(0,1,2개정도)
    sum_df['샘플값0'] = train.loc[0].values
    sum_df['샘플값1'] = train.loc[1].values
    sum_df['샘플값2'] = train.loc[2].values

    return sum_df

summary_df = summary_feature_info( train )
summary_df
```





  <div id="df-2159fc69-8d60-41fe-9263-0bce808ea7a5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_name</th>
      <th>type</th>
      <th>결측치수</th>
      <th>고유값수</th>
      <th>샘플값0</th>
      <th>샘플값1</th>
      <th>샘플값2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Time</td>
      <td>float64</td>
      <td>0</td>
      <td>36845</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V1</td>
      <td>float64</td>
      <td>0</td>
      <td>217723</td>
      <td>2.074329</td>
      <td>1.998827</td>
      <td>0.091535</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V2</td>
      <td>float64</td>
      <td>0</td>
      <td>217729</td>
      <td>-0.129425</td>
      <td>-1.250891</td>
      <td>1.004517</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V3</td>
      <td>float64</td>
      <td>0</td>
      <td>217700</td>
      <td>-1.137418</td>
      <td>-0.520969</td>
      <td>-0.223445</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V4</td>
      <td>float64</td>
      <td>0</td>
      <td>217715</td>
      <td>0.412846</td>
      <td>-0.894539</td>
      <td>-0.435249</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V5</td>
      <td>float64</td>
      <td>0</td>
      <td>217661</td>
      <td>-0.192638</td>
      <td>-1.122528</td>
      <td>0.667548</td>
    </tr>
    <tr>
      <th>6</th>
      <td>V6</td>
      <td>float64</td>
      <td>0</td>
      <td>217594</td>
      <td>-1.210144</td>
      <td>-0.270866</td>
      <td>-0.988351</td>
    </tr>
    <tr>
      <th>7</th>
      <td>V7</td>
      <td>float64</td>
      <td>0</td>
      <td>217735</td>
      <td>0.110697</td>
      <td>-1.029289</td>
      <td>0.948146</td>
    </tr>
    <tr>
      <th>8</th>
      <td>V8</td>
      <td>float64</td>
      <td>0</td>
      <td>217679</td>
      <td>-0.263477</td>
      <td>0.050198</td>
      <td>-0.084789</td>
    </tr>
    <tr>
      <th>9</th>
      <td>V9</td>
      <td>float64</td>
      <td>0</td>
      <td>217681</td>
      <td>0.742144</td>
      <td>-0.109948</td>
      <td>-0.042027</td>
    </tr>
    <tr>
      <th>10</th>
      <td>V10</td>
      <td>float64</td>
      <td>0</td>
      <td>217722</td>
      <td>0.108782</td>
      <td>0.908773</td>
      <td>-0.818383</td>
    </tr>
    <tr>
      <th>11</th>
      <td>V11</td>
      <td>float64</td>
      <td>0</td>
      <td>217680</td>
      <td>-1.070243</td>
      <td>0.836798</td>
      <td>-0.376512</td>
    </tr>
    <tr>
      <th>12</th>
      <td>V12</td>
      <td>float64</td>
      <td>0</td>
      <td>217739</td>
      <td>-0.234910</td>
      <td>-0.056580</td>
      <td>-0.226546</td>
    </tr>
    <tr>
      <th>13</th>
      <td>V13</td>
      <td>float64</td>
      <td>0</td>
      <td>217760</td>
      <td>-1.099360</td>
      <td>-0.120990</td>
      <td>-0.552869</td>
    </tr>
    <tr>
      <th>14</th>
      <td>V14</td>
      <td>float64</td>
      <td>0</td>
      <td>217726</td>
      <td>0.502467</td>
      <td>-0.144028</td>
      <td>-0.886466</td>
    </tr>
    <tr>
      <th>15</th>
      <td>V15</td>
      <td>float64</td>
      <td>0</td>
      <td>217721</td>
      <td>0.169318</td>
      <td>-0.039582</td>
      <td>-0.180890</td>
    </tr>
    <tr>
      <th>16</th>
      <td>V16</td>
      <td>float64</td>
      <td>0</td>
      <td>217749</td>
      <td>0.065688</td>
      <td>1.653057</td>
      <td>0.230286</td>
    </tr>
    <tr>
      <th>17</th>
      <td>V17</td>
      <td>float64</td>
      <td>0</td>
      <td>217741</td>
      <td>-0.306957</td>
      <td>-0.253599</td>
      <td>0.590579</td>
    </tr>
    <tr>
      <th>18</th>
      <td>V18</td>
      <td>float64</td>
      <td>0</td>
      <td>217726</td>
      <td>-0.323800</td>
      <td>-0.814354</td>
      <td>-0.321590</td>
    </tr>
    <tr>
      <th>19</th>
      <td>V19</td>
      <td>float64</td>
      <td>0</td>
      <td>217750</td>
      <td>0.103348</td>
      <td>0.716784</td>
      <td>-0.433959</td>
    </tr>
    <tr>
      <th>20</th>
      <td>V20</td>
      <td>float64</td>
      <td>0</td>
      <td>217742</td>
      <td>-0.292969</td>
      <td>0.065717</td>
      <td>-0.021375</td>
    </tr>
    <tr>
      <th>21</th>
      <td>V21</td>
      <td>float64</td>
      <td>0</td>
      <td>217781</td>
      <td>-0.334701</td>
      <td>0.054848</td>
      <td>-0.326725</td>
    </tr>
    <tr>
      <th>22</th>
      <td>V22</td>
      <td>float64</td>
      <td>0</td>
      <td>217752</td>
      <td>-0.887840</td>
      <td>-0.038367</td>
      <td>-0.803736</td>
    </tr>
    <tr>
      <th>23</th>
      <td>V23</td>
      <td>float64</td>
      <td>0</td>
      <td>217783</td>
      <td>0.336701</td>
      <td>0.133518</td>
      <td>0.154495</td>
    </tr>
    <tr>
      <th>24</th>
      <td>V24</td>
      <td>float64</td>
      <td>0</td>
      <td>217687</td>
      <td>-0.110835</td>
      <td>-0.461928</td>
      <td>0.951233</td>
    </tr>
    <tr>
      <th>25</th>
      <td>V25</td>
      <td>float64</td>
      <td>0</td>
      <td>217761</td>
      <td>-0.291459</td>
      <td>-0.465491</td>
      <td>-0.506919</td>
    </tr>
    <tr>
      <th>26</th>
      <td>V26</td>
      <td>float64</td>
      <td>0</td>
      <td>217771</td>
      <td>0.207733</td>
      <td>-0.464655</td>
      <td>0.085046</td>
    </tr>
    <tr>
      <th>27</th>
      <td>V27</td>
      <td>float64</td>
      <td>0</td>
      <td>217754</td>
      <td>-0.076576</td>
      <td>-0.009413</td>
      <td>0.224458</td>
    </tr>
    <tr>
      <th>28</th>
      <td>V28</td>
      <td>float64</td>
      <td>0</td>
      <td>217758</td>
      <td>-0.059577</td>
      <td>-0.038238</td>
      <td>0.087356</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Amount</td>
      <td>float64</td>
      <td>0</td>
      <td>19585</td>
      <td>1.980000</td>
      <td>84.000000</td>
      <td>2.690000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Class</td>
      <td>int64</td>
      <td>0</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2159fc69-8d60-41fe-9263-0bce808ea7a5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2159fc69-8d60-41fe-9263-0bce808ea7a5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2159fc69-8d60-41fe-9263-0bce808ea7a5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




- 데이터는 전부 수치형이다 -> 인코딩 필요 X
- LGBM을 사용하지만 스케일링을 적용하는 것이 성능에 도움이 된다는 것을 확인하였다 -> MinMaxscaler로 스케일링

## 결측치 확인


```python
# 결측치 확인 -> 결측치 없다
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 219129 entries, 0 to 219128
    Data columns (total 31 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    219129 non-null  float64
     1   V1      219129 non-null  float64
     2   V2      219129 non-null  float64
     3   V3      219129 non-null  float64
     4   V4      219129 non-null  float64
     5   V5      219129 non-null  float64
     6   V6      219129 non-null  float64
     7   V7      219129 non-null  float64
     8   V8      219129 non-null  float64
     9   V9      219129 non-null  float64
     10  V10     219129 non-null  float64
     11  V11     219129 non-null  float64
     12  V12     219129 non-null  float64
     13  V13     219129 non-null  float64
     14  V14     219129 non-null  float64
     15  V15     219129 non-null  float64
     16  V16     219129 non-null  float64
     17  V17     219129 non-null  float64
     18  V18     219129 non-null  float64
     19  V19     219129 non-null  float64
     20  V20     219129 non-null  float64
     21  V21     219129 non-null  float64
     22  V22     219129 non-null  float64
     23  V23     219129 non-null  float64
     24  V24     219129 non-null  float64
     25  V25     219129 non-null  float64
     26  V26     219129 non-null  float64
     27  V27     219129 non-null  float64
     28  V28     219129 non-null  float64
     29  Amount  219129 non-null  float64
     30  Class   219129 non-null  int64  
    dtypes: float64(30), int64(1)
    memory usage: 61.6 MB



```python
# 결측치 확인 -> 결측치 없다
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 146087 entries, 219129 to 365215
    Data columns (total 30 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    146087 non-null  float64
     1   V1      146087 non-null  float64
     2   V2      146087 non-null  float64
     3   V3      146087 non-null  float64
     4   V4      146087 non-null  float64
     5   V5      146087 non-null  float64
     6   V6      146087 non-null  float64
     7   V7      146087 non-null  float64
     8   V8      146087 non-null  float64
     9   V9      146087 non-null  float64
     10  V10     146087 non-null  float64
     11  V11     146087 non-null  float64
     12  V12     146087 non-null  float64
     13  V13     146087 non-null  float64
     14  V14     146087 non-null  float64
     15  V15     146087 non-null  float64
     16  V16     146087 non-null  float64
     17  V17     146087 non-null  float64
     18  V18     146087 non-null  float64
     19  V19     146087 non-null  float64
     20  V20     146087 non-null  float64
     21  V21     146087 non-null  float64
     22  V22     146087 non-null  float64
     23  V23     146087 non-null  float64
     24  V24     146087 non-null  float64
     25  V25     146087 non-null  float64
     26  V26     146087 non-null  float64
     27  V27     146087 non-null  float64
     28  V28     146087 non-null  float64
     29  Amount  146087 non-null  float64
    dtypes: float64(30)
    memory usage: 34.6 MB



```python
# 결측치 확인 -> 결측치 없다
orig_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    284807 non-null  float64
     1   V1      284807 non-null  float64
     2   V2      284807 non-null  float64
     3   V3      284807 non-null  float64
     4   V4      284807 non-null  float64
     5   V5      284807 non-null  float64
     6   V6      284807 non-null  float64
     7   V7      284807 non-null  float64
     8   V8      284807 non-null  float64
     9   V9      284807 non-null  float64
     10  V10     284807 non-null  float64
     11  V11     284807 non-null  float64
     12  V12     284807 non-null  float64
     13  V13     284807 non-null  float64
     14  V14     284807 non-null  float64
     15  V15     284807 non-null  float64
     16  V16     284807 non-null  float64
     17  V17     284807 non-null  float64
     18  V18     284807 non-null  float64
     19  V19     284807 non-null  float64
     20  V20     284807 non-null  float64
     21  V21     284807 non-null  float64
     22  V22     284807 non-null  float64
     23  V23     284807 non-null  float64
     24  V24     284807 non-null  float64
     25  V25     284807 non-null  float64
     26  V26     284807 non-null  float64
     27  V27     284807 non-null  float64
     28  V28     284807 non-null  float64
     29  Amount  284807 non-null  float64
     30  Class   284807 non-null  int64  
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB


## 데이터 시각화를 통한 분석

### 타겟(정답) 값 분포


```python
# 향후 시각화 자료에 비율을 표기하는 함수 생성
def show_text_percent_by_targert( ax, total_count, is_show=True,  ):

  persents = list()

  for patch in ax.patches: 
    w = patch.get_width() 
    h = patch.get_height() 
    p = h/total_count * 100 
    persents.append( p )
    
    l = patch.get_x() 
    ax.text(  x=l + w/2,
              y=h + total_count*0.005,
              s=f'{p:1.1f}%',
              ha='center' 
            )
  if is_show:
    if len(persents) == 2:
      print( '타겟값간 비율', persents[0]/persents[1] )
    elif len(persents) == 4:
      print( persents[0]/persents[2], persents[1]/persents[3] )
  pass

# train의 정답 비율
ax = sns.countplot( data=train, x='Class');
show_text_percent_by_targert( ax, train.shape[0] )
ax.set_title('Target Value Distribution: train')
```

    타겟값간 비율 466.226012793177





    Text(0.5, 1.0, 'Target Value Distribution: train')




![png](%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B24%E1%84%87%E1%85%A5%E1%86%AB%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC_files/%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B24%E1%84%87%E1%85%A5%E1%86%AB%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC_18_2.png)



```python
# orig_train의 정답 비율
ax = sns.countplot( data=orig_train, x='Class');
show_text_percent_by_targert( ax, orig_train.shape[0] )
ax.set_title('Target Value Distribution: orig_train')
```

    타겟값간 비율 577.8760162601625





    Text(0.5, 1.0, 'Target Value Distribution: orig_train')




![png](%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B24%E1%84%87%E1%85%A5%E1%86%AB%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC_files/%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B24%E1%84%87%E1%85%A5%E1%86%AB%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC_19_2.png)


- 정답의 비율이 0.2%로 매우 적다
- train과 orig_train의 정답 비율은 같다!

### 각 컬럼별 분포도 확인


```python
features = [c for c in train.columns if c not in ['id', 'Class']]
print(features)
```

    ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']



```python
fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(15, 45))

for n in range(30):
    m = n%3
    l = int(n/3)
    sns.kdeplot(ax = axes[l,m], x = features[n],data = train)
```


![png](%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B24%E1%84%87%E1%85%A5%E1%86%AB%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC_files/%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B24%E1%84%87%E1%85%A5%E1%86%AB%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC_23_0.png)


### 각 컬럼별 train, test, orig_train 데이터 분포도 확인


```python
fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(15, 45))
for ax, n in zip(axes.flat, features):
    sns.kdeplot(train[n], color='r', label='train', ax=ax)
    sns.kdeplot(test[n], color='b', label='test', ax=ax)
    sns.kdeplot(orig_train[n], color='g', label='orig_train', ax=ax)
    ax.set_title(n)
    ax.legend()
plt.tight_layout()
plt.show()
```


![png](%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B24%E1%84%87%E1%85%A5%E1%86%AB%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC_files/%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B24%E1%84%87%E1%85%A5%E1%86%AB%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC_25_0.png)


### 분석결과

- train 데이터에서 'Time'의 분포도를 확인했을 때, 정규분포를 이루고 있어 학습에 도움이 될 것이라 예상하였다
- 하지만 train, test, orig_train과 비교했더니 'Time'만 분포가 매우 달랐다 => 'Time' 컬럼만 제거 후 orig_train 사용하겠다

# 피처엔지니어링


```python
# train에 original 데이터 병합
train = pd.concat([train,orig_train]).reset_index(drop=True)
```


```python
train.shape, test.shape
```




    ((503936, 31), (146087, 30))



- 총 503936개의 train 데이터와 146087개의 test 데이터를 사용하겠다!

## 스케일링


```python
X = train.iloc[ :, :-1]
y = train.iloc[ :, -1: ]

# 정답을 제외한 피처들
features = [c for c in train.columns if c not in ['id', 'Class']]

# 스케일링
alpha_enc = MinMaxScaler()
temp_alpha = pd.concat( [ train.iloc[:,:-1], test ] )
alpha_enc.fit( temp_alpha )
enc_nom_etc_train = alpha_enc.transform( train.iloc[:,:-1]  )
enc_nom_etc_test  = alpha_enc.transform( test )

train=pd.DataFrame( enc_nom_etc_train)
test=pd.DataFrame( enc_nom_etc_test)

train.columns = features
test.columns = features

# train에 다시 정답 컬럼 추가
train['Class'] = y
```


```python
# 스케일링 잘 되었다!
train.head()
```





  <div id="df-df43b57c-74b2-459f-9b8c-ba829aa2f5cb">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.993534</td>
      <td>0.765893</td>
      <td>0.817704</td>
      <td>0.270231</td>
      <td>0.764419</td>
      <td>0.250853</td>
      <td>0.266030</td>
      <td>0.782559</td>
      <td>0.488345</td>
      <td>...</td>
      <td>0.556084</td>
      <td>0.468613</td>
      <td>0.670434</td>
      <td>0.367299</td>
      <td>0.561546</td>
      <td>0.459381</td>
      <td>0.415098</td>
      <td>0.311915</td>
      <td>0.000077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.992251</td>
      <td>0.754060</td>
      <td>0.828386</td>
      <td>0.212276</td>
      <td>0.758159</td>
      <td>0.260296</td>
      <td>0.259085</td>
      <td>0.785924</td>
      <td>0.458992</td>
      <td>...</td>
      <td>0.562364</td>
      <td>0.508241</td>
      <td>0.667416</td>
      <td>0.319990</td>
      <td>0.551777</td>
      <td>0.349548</td>
      <td>0.416337</td>
      <td>0.312348</td>
      <td>0.003270</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.959849</td>
      <td>0.777858</td>
      <td>0.833542</td>
      <td>0.232636</td>
      <td>0.770210</td>
      <td>0.253083</td>
      <td>0.271132</td>
      <td>0.784476</td>
      <td>0.461332</td>
      <td>...</td>
      <td>0.556213</td>
      <td>0.472537</td>
      <td>0.667728</td>
      <td>0.510412</td>
      <td>0.549452</td>
      <td>0.439340</td>
      <td>0.420654</td>
      <td>0.314897</td>
      <td>0.000105</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.991926</td>
      <td>0.765307</td>
      <td>0.818972</td>
      <td>0.257255</td>
      <td>0.764267</td>
      <td>0.256496</td>
      <td>0.264820</td>
      <td>0.785006</td>
      <td>0.493295</td>
      <td>...</td>
      <td>0.559940</td>
      <td>0.506309</td>
      <td>0.667924</td>
      <td>0.376448</td>
      <td>0.577951</td>
      <td>0.409743</td>
      <td>0.415444</td>
      <td>0.311625</td>
      <td>0.000039</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.975723</td>
      <td>0.765445</td>
      <td>0.858272</td>
      <td>0.307071</td>
      <td>0.761431</td>
      <td>0.274070</td>
      <td>0.259637</td>
      <td>0.791492</td>
      <td>0.486641</td>
      <td>...</td>
      <td>0.563078</td>
      <td>0.538437</td>
      <td>0.665848</td>
      <td>0.346820</td>
      <td>0.592379</td>
      <td>0.384149</td>
      <td>0.418511</td>
      <td>0.313551</td>
      <td>0.000039</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-df43b57c-74b2-459f-9b8c-ba829aa2f5cb')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-df43b57c-74b2-459f-9b8c-ba829aa2f5cb button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-df43b57c-74b2-459f-9b8c-ba829aa2f5cb');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Time 컬럼 제거


```python
# Time 제거
train = train.iloc[:,1:]
test = test.iloc[:,1:]
```


```python
train.head(1)
```





  <div id="df-ea3012a8-aff4-4270-91d7-dc90e9affb78">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.993534</td>
      <td>0.765893</td>
      <td>0.817704</td>
      <td>0.270231</td>
      <td>0.764419</td>
      <td>0.250853</td>
      <td>0.26603</td>
      <td>0.782559</td>
      <td>0.488345</td>
      <td>0.510973</td>
      <td>...</td>
      <td>0.556084</td>
      <td>0.468613</td>
      <td>0.670434</td>
      <td>0.367299</td>
      <td>0.561546</td>
      <td>0.459381</td>
      <td>0.415098</td>
      <td>0.311915</td>
      <td>0.000077</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 30 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ea3012a8-aff4-4270-91d7-dc90e9affb78')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ea3012a8-aff4-4270-91d7-dc90e9affb78 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ea3012a8-aff4-4270-91d7-dc90e9affb78');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
test.head(1)
```





  <div id="df-98ae9661-5c9f-49df-9c46-5f8860de4ce4">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.994234</td>
      <td>0.759959</td>
      <td>0.814791</td>
      <td>0.221536</td>
      <td>0.761399</td>
      <td>0.255331</td>
      <td>0.259725</td>
      <td>0.785138</td>
      <td>0.454852</td>
      <td>0.513286</td>
      <td>...</td>
      <td>0.580983</td>
      <td>0.565376</td>
      <td>0.541871</td>
      <td>0.665995</td>
      <td>0.308545</td>
      <td>0.570999</td>
      <td>0.411582</td>
      <td>0.416598</td>
      <td>0.312679</td>
      <td>0.001166</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 29 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-98ae9661-5c9f-49df-9c46-5f8860de4ce4')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-98ae9661-5c9f-49df-9c46-5f8860de4ce4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-98ae9661-5c9f-49df-9c46-5f8860de4ce4');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# Optuna를 이용해 최적의 파라미터 찾기


```python
# optuna 설치
!pip install optuna
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting optuna
      Downloading optuna-3.1.0-py3-none-any.whl (365 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m365.3/365.3 KB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting cmaes>=0.9.1
      Downloading cmaes-0.9.1-py3-none-any.whl (21 kB)
    Requirement already satisfied: numpy in /usr/local/lib/python3.8/dist-packages (from optuna) (1.21.6)
    Requirement already satisfied: sqlalchemy>=1.3.0 in /usr/local/lib/python3.8/dist-packages (from optuna) (1.4.46)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.8/dist-packages (from optuna) (23.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.8/dist-packages (from optuna) (4.64.1)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.8/dist-packages (from optuna) (6.0)
    Collecting colorlog
      Downloading colorlog-6.7.0-py2.py3-none-any.whl (11 kB)
    Collecting alembic>=1.5.0
      Downloading alembic-1.9.4-py3-none-any.whl (210 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m210.5/210.5 KB[0m [31m24.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: importlib-metadata in /usr/local/lib/python3.8/dist-packages (from alembic>=1.5.0->optuna) (6.0.0)
    Collecting Mako
      Downloading Mako-1.2.4-py3-none-any.whl (78 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m78.7/78.7 KB[0m [31m9.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: importlib-resources in /usr/local/lib/python3.8/dist-packages (from alembic>=1.5.0->optuna) (5.10.2)
    Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.8/dist-packages (from sqlalchemy>=1.3.0->optuna) (2.0.2)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.8/dist-packages (from importlib-metadata->alembic>=1.5.0->optuna) (3.12.1)
    Requirement already satisfied: MarkupSafe>=0.9.2 in /usr/local/lib/python3.8/dist-packages (from Mako->alembic>=1.5.0->optuna) (2.0.1)
    Installing collected packages: Mako, colorlog, cmaes, alembic, optuna
    Successfully installed Mako-1.2.4 alembic-1.9.4 cmaes-0.9.1 colorlog-6.7.0 optuna-3.1.0



```python
# 모듈불러오기
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgbm
from tqdm import tqdm
```


```python
# 함수 생성
def opt(trial):
    # 파라미터 범위 지정
    params = {
        'scale_pos_weight':trial.suggest_int('scale_pos_weight', 1, 3),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-12, 2, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 5, 25.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 35, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.65, 0.85),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.65),
        'bagging_freq': trial.suggest_int('bagging_freq', 4, 9),
         'min_child_samples': trial.suggest_int('min_child_samples', 40, 90),
         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 90, 150),
        "max_depth": trial.suggest_int("max_depth", 6, 12),

        'num_iterations':10000,
        'learning_rate':0.1
    }

    # KFold 사용
    n=3
    cv = StratifiedKFold(n,shuffle=True, random_state=42)

    # 점수를 담을 그릇 생성
    scores = []

    # Time을 제거했기 때문에 피처들 다시 선언
    features = [c for c in train.columns if c not in ['id', 'Class']]

    for i,(train_idx,val_idx) in enumerate(cv.split(train[features],train['Class'])):
        X_train, y_train = train.loc[train_idx, features],train.loc[train_idx, 'Class']
        X_val, y_val = train.loc[val_idx, features],train.loc[val_idx, 'Class']

        # 모델 생성
        model = lgbm.LGBMClassifier(**params)
        # 훈련
        model.fit(X_train, y_train, eval_set = [(X_val,y_val)], early_stopping_rounds=50, verbose=500)
        # 예측
        y_pred = model.predict_proba(X_val)[:,1]
        # 평가
        score = roc_auc_score(y_val,y_pred)
        # 점수를 그릇에 담기
        scores.append(score)

    return np.mean(scores)
```


```python
# optuna로 최적의 파라미터 찾는 코드
'''
study = optuna.create_study(direction='maximize', sampler = TPESampler())
study.optimize(func=opt, n_trials=100)
study.best_params
'''
```




    "\nstudy = optuna.create_study(direction='maximize', sampler = TPESampler())\nstudy.optimize(func=opt, n_trials=100)\nstudy.best_params\n"



- 최적의 파라미터 : {
      'objective': 'binary',
      'metric': 'auc',
      'num_iterations':300,
      'learning_rate':0.05,
      'scale_pos_weight': 1,
      'lambda_l1': 8.629495026974812e-12,
      'lambda_l2': 9.464608889367986,
      'num_leaves': 39,
      'feature_fraction': 0.6816416268789407,
      'bagging_fraction': 0.6444301272381365,
      'bagging_freq': 8,
      'min_child_samples': 83,
      'min_data_in_leaf': 102,
      'max_depth': 7
}

# 알고리즘 생성, 예측


```python
# KFold 사용
n=5
cv = StratifiedKFold(n,shuffle=True, random_state=42)

# 피처 목록
features = [c for c in train.columns if c not in ['id', 'Class']]

# 점수를 담을 그릇 생성
all_scores = []
# 최종 예측값을 담을 그릇 생성
test_preds = []

for i,(train_idx,val_idx) in enumerate(cv.split(train[features],train['Class'])):
    X_train, y_train = train.loc[train_idx, features],train.loc[train_idx, 'Class']
    X_val, y_val = train.loc[val_idx, features],train.loc[val_idx, 'Class']
    
    # optuna를 이용해 찾은 파라미터
    params={
          'objective': 'binary',
          'metric': 'auc',
          'num_iterations':300,
          'learning_rate':0.05,
          'scale_pos_weight': 1,
          'lambda_l1': 8.629495026974812e-12,
          'lambda_l2': 9.464608889367986,
          'num_leaves': 39,
          'feature_fraction': 0.6816416268789407,
          'bagging_fraction': 0.6444301272381365,
          'bagging_freq': 8,
          'min_child_samples': 83,
          'min_data_in_leaf': 102,
          'max_depth': 7
    }
    
    # 모델 생성
    model = lgbm.LGBMClassifier(**params)
    # 훈련
    model.fit(X_train, y_train, eval_set = [(X_val,y_val)], early_stopping_rounds=50, verbose=500)
    # 예측
    y_pred = model.predict_proba(X_val)[:,1]
    # 평가
    score = roc_auc_score(y_val,y_pred)
    # 점수를 그릇에 담기
    all_scores.append(score)
    # 최종 예측
    test_pred = model.predict_proba(test)[:,1]
    # 최종 예측값을 그릇에 담기
    test_preds.append(test_pred)

    print(f'=== Fold {i} ROC AUC Score {score} ===')

print(f'=== Average ROC AUC Score {np.mean(all_scores)} ===')
```

    /usr/local/lib/python3.8/dist-packages/lightgbm/engine.py:118: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))


    Training until validation scores don't improve for 50 rounds.
    Early stopping, best iteration is:
    [122]	valid_0's auc: 0.897236
    === Fold 0 ROC AUC Score 0.8972356705581066 ===


    /usr/local/lib/python3.8/dist-packages/lightgbm/engine.py:118: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))


    Training until validation scores don't improve for 50 rounds.
    Early stopping, best iteration is:
    [32]	valid_0's auc: 0.89835
    === Fold 1 ROC AUC Score 0.8983501810063456 ===


    /usr/local/lib/python3.8/dist-packages/lightgbm/engine.py:118: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))


    Training until validation scores don't improve for 50 rounds.
    Early stopping, best iteration is:
    [170]	valid_0's auc: 0.889852
    === Fold 2 ROC AUC Score 0.889851555122024 ===


    /usr/local/lib/python3.8/dist-packages/lightgbm/engine.py:118: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))


    Training until validation scores don't improve for 50 rounds.
    Early stopping, best iteration is:
    [70]	valid_0's auc: 0.906923
    === Fold 3 ROC AUC Score 0.9069226643139984 ===


    /usr/local/lib/python3.8/dist-packages/lightgbm/engine.py:118: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))


    Training until validation scores don't improve for 50 rounds.
    Early stopping, best iteration is:
    [158]	valid_0's auc: 0.896967
    === Fold 4 ROC AUC Score 0.8969670564309029 ===
    === Average ROC AUC Score 0.8978654254862756 ===


# 제출


```python
submission['Class'] = np.array(test_preds).mean(axis=0)
submission.to_csv('submission.csv', index=False)
```


```python
!jupyter nbconvert --to markdown "/content/drive/MyDrive/cloud_ai/3.머신러닝/wine_test.ipynb"
```
