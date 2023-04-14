**We use KuaiRec as our dataset**

download Kuaishou Rec dataset
```
!wget https://chongming.myds.me:61364/data/KuaiRec.zip --no-check-certificate
!unzip KuaiRec.zip
```

Dataset description
```
  KuaiRec
  ├── data
  │   ├── big_matrix.csv          
  │   ├── small_matrix.csv
  │   ├── social_network.csv
  │   ├── user_features.csv
  │   ├── item_daily_features.csv
  │   └── item_categories.csv
  ```
  
  Citation
  ```
  @inproceedings{gao2022kuairec,
  author = {Chongming Gao and Shijun Li and Wenqiang Lei and Jiawei Chen and Biao Li and Peng Jiang and Xiangnan He and Jiaxin Mao and Tat-Seng Chua},
  title = {KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems},
  year = {2022},
  url = {https://doi.org/10.1145/3511808.3557220},
  doi = {10.1145/3511808.3557220},
  booktitle = {Proceedings of the 31st ACM International Conference on Information and Knowledge Management},
  numpages = {11},
  location = {Atlanta, GA, USA},
  series = {CIKM '22}
}
```
