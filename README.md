# predictive-modelling-fognets

This repo contains some of the functions I have written during the Data Science competition ["From Fog Nets to Neural Nets"](https://www.drivendata.org/competitions/9/) hosted by Dar Si Hmad on drivendata.org.

An important aspect of this competition was the sequential structure of the train/test split (see figure below).

![alt text][train/test tsplit]
[train/test tsplit]: https://lh3.googleusercontent.com/ayxFP9aRJiqpYUG8W5iEmm-xex8k9yku-aOzUMx3Y5Y2x7ythy6xjx7WVp6Y98gXjkzeBonbYkf574IUcef1Ttx8xifh3ntloWqjNoctr5V_wMieAAutJxh2vibTSqKI8mCqDjRyl_FOpEalYvc8QDgwAYfo6kWUFuXVwdJmZxghuV5Df5dr2pUaBVN61L-hld941tM7CtfNjCwROtUswZDo8E3Ywle2pWJhFRHCqsXn4z74CHo-W3hqJmMFAvZKtf8aUwMbQl0e50Yomczck_i7hYsl0w8XJMwhcRyFllO5r6NyJJUs2wEk206dhvTdPYBEFYwrB8D-h5POuoRuBpa02YTXAc8_Sx1REHwDvCAd4KMubiECjsMzT3TNVe1z8elC0yoosyiaJYnBo7Jlup9a8GtWA4VnWZiJn-q5fu-an50NQ5YV5P1I08mJwJGVVdEogaNL9xVHm9KKiKJ_QPCaX1mg5qV_IY1vkZzW7fT7-Xk0eT3szdeD9Ivv4NQwDdB-dQUdakfu3Zp6TyHV9p9OIJPDTYUrWk7XPUdj3BYO2Y5CBj-WB1FyT6vyBOaJJThR2VsuZH6vOIMeuh0ahx90sQHXreY=w1196-h433-no

For this reason I have writted a set of functions to work sequentially on several common tasks such as model training, imputation and exploration of models' performances without polluting the process with data from the future.
