# [2025 Knowledge Based Systems] "HiLINK: Hierarchical Linking of Context-Aware Knowledge Prediction and Prompt Tuning for Bilingual Knowledge-Based Visual Question Answering

## 📖 Overview

HiLINK is a novel framework designed to enhance Knowledge-Based Visual Question Answering (KBVQA) performance in multilingual environments, particularly addressing the challenges in low-resource languages. It aims to overcome limitations of existing KBVQA models, such as the need for retraining with new knowledge and difficulties in aligning diverse embedding spaces.

![fig_3_HiLINK](https://github.com/user-attachments/assets/a4957545-20d0-4ac8-9832-8bc1d5f6a2e9)

## ✨ Key Features

* **End-to-End Knowledge Transfer via Link-Tuning**: HiLINK proposes `Link-Tuning` to directly learn relationships between triplet knowledge components within prompts, eliminating the need for a separate Knowledge Graph Embedding (KGE) training network. This simplifies the training process and removes the necessity for continuous retraining when new knowledge is added.
  
![fig_5_LinkTuning](https://github.com/user-attachments/assets/fbf60f1c-b8a4-4f33-a0f0-f797b2094347)
  
* **Context-Aware Triplet Prediction (HK-TriNet & HK-TriNet+)**: We introduce `HK-TriNet` and `HK-TriNet+` to improve triplet prediction by leveraging interdependencies among triplet elements. `HK-TriNet+` uses a soft ensemble mechanism with cascading and back-cascading structures to model complex interactions and capture bidirectional relationships effectively.
  
![fig_6_HK_TriNetpng](https://github.com/user-attachments/assets/7945ad67-2f53-4878-91b3-8b840ad8fb20)

* **Efficient Bilingual Learning Strategy**: The framework empirically validates a frozen training approach for the image encoder while keeping the text encoder trainable. This balances computational efficiency with adaptability, allowing the model to preserve general visual features and focus on language-specific semantic representations, leading to more effective knowledge transfer across language.
  
![fig_8_TranableSetting](https://github.com/user-attachments/assets/cb580524-abd8-4f4b-adc1-2f54182a67bc)

## 🚀 Performance

HiLINK demonstrates outstanding performance on the BOK-VQA dataset, outperforming the GEL-VQA method.

![image](https://github.com/user-attachments/assets/2ebe96e0-f47e-4cf5-be6b-aa02cbcf1312)

* **Bilingual Environment** : **+19.40%** improvement over GEL-VQA.
* **English Environment**   : **+12.01%** improvement over GEL-VQA.
* **Korean Environment**    : **+11.30%** improvement over GEL-VQA.

Notably, the bilingual configuration acheived better performance than monolingual ones, demonstrating the efficacy of cross-language combinations.


## 📊 Datasets
This research primarily utilizes the BOK-VQA dataset.
Composition : It consists of 17,836 images and 282,533 knowledge triplets.
Languages   : Includes 17,836 English queries and 17,836 Korean queries corresponding to each image. The bilingual environment combines these, totaling 35,672 image-multilingual query pairs.
Access      : The BOK-VQA dataset requires access permission for researchers. Please refer to the original paper or relevant BOK-VQA resources for download and access procedures.

You can download the image files via [G-drive](https://drive.google.com/file/d/1SpOntv2ZIwyNW-JghUc7myJkC9PLs4_H/view) or [AI-hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71357)    
After the download is complete, place the image directory inside the data directory.
Your directory structure will then look like this:

```
data/
├── image/
│   ├── 121100228220707140119.jpg
│   ├── 121100228220707140304.jpg
│   ├── ...
│   └── 121100520220830104341.jpg
├── all_triple.csv
├── BOKVQA_data_en.csv
├── BOKVQA_data_ko.csv
├── BOKVQA_data_test_en.csv
└── BOKVQA_data_test_ko.csv
```

## ⚙️ Usage
You can find the preprocessed CSV data in the data directory.

* all_triple.csv : The entire knowledge base consisting of 282,533 triples.
* BOKVQA_data_en.csv: English BOKVQA data for training.
* BOKVQA_data_test_en.csv: English BOKVQA data for testing.
* BOKVQA_data_ko.csv: Korean BOKVQA data for training.
* BOKVQA_data_test_ko.csv: Korean BOKVQA data for testing.

### Installation all requirements.

```
pip install -r requirements.txt
```
### Train the model

```
cd HiLINK_Plus
```
* To train the HiLINK_TriNet+ model, use the following command:
```
bash run.sh
```

### Test  the model

```
cd HiLINK_Plus
```
* To test the HiLINK_TriNet+ model, use the following command:
```
bash test_run.sh
```

## 📝 Citation
If this research is helpful to your work, please cite it using the following BibTeX format:
```
@article{jeong2025hilink,
  title={HiLINK: Hierarchical linking of context-aware knowledge prediction and prompt tuning for bilingual knowledge-based visual question answering},
  author={Jeong, Hyeonki and Kim, Taehyeong and Shin, Wooseok and Han, Sung Won},
  journal={Knowledge-Based Systems},
  pages={113556},
  year={2025},
  publisher={Elsevier}
}
```

## 📧 Contact
For any questions or suggestions regarding this research or code, please feel free to contact:

Hyeonki Jeong: 
gusrl1210@korea.ac.kr
