# VideoTextClassification
동영상은 화상(움직이는 이미지)와 음향으로 구성되어 있다.

화상(움직이는 이미지)은 일련의 이미지들로 분할될 수 있으며, 일부 동영상에서 음향은 말(speech)을 의미하기도 한다.

영상 콘텐츠의 공급과 수요가 급증하고 자동 자막 혹은 STT 연구가 고도화되고 있는 상황에서, 본 연구는 video와 text의 문맥에 대한 이해를 기반으로 하는 video classification의 아키텍쳐를 제안한다.

이를 위해 트랜스포머(Encoder only)기반의 비디오 임베딩 모델을 새로이 구성하여 이미지 시퀀스의 문맥을 파악한다.

본 연구는 나아가 Same Topic Prediction의 관점에서 Temporal Video Segmentation Task에도 적용될 수 있다.

![image](https://user-images.githubusercontent.com/61833149/173534098-7a419d72-5758-4d27-be62-b56d30548a58.png)

![image](https://user-images.githubusercontent.com/61833149/173534459-09868e3b-2a97-4122-a26b-6525e9451d65.png)
