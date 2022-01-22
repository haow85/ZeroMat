# ZeroMat

ZeroMat is a cold-start solver for recommender systems invented by Hao Wang and published at ICISCAE 2021.

The paper can be found here : https://arxiv.org/abs/2112.03084

Please modify Line 372 for the input location of MovieLens 1M dataset's rating values if necessary.

ZeroMat is really important because:
    
    1. It is the first algorithm that solves cold start problem in recommender systems with no input data.
    2. The experiments on the paper demonstrate that human cultural tastes (or, user item rating data) follows a predictable distribution after evolution for sometime.
    3. Because of observations in 2, our public culture is pre-determined. Human cultural evolution converges to Zipf's distribution and the details have no relation with history.

To execute ZeroMat, run the following commands :

python3 process_data.py
python3 compute_cold_start.py

A sample plot of results would look like the following :

![MAE](https://user-images.githubusercontent.com/20984794/150656430-66836cfe-3223-4b8e-9ecb-5a858728ea34.png)


![DME](https://user-images.githubusercontent.com/20984794/150656435-236be9f8-9d27-4d41-a2e0-63e43ff1d3f8.png)
