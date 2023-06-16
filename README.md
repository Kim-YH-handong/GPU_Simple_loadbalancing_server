# GPU_Simple_loadbalancing_server
GPU가 없는 클라이언트들이 TCP 연결을 통해 GPU 서버에서 GPU를 사용할 수 있게 만든 코드입니다.

또한, GPU를 사용하는 과정에서 메인 서버를 추가하여 Load balancing 역할을 하게 해주었습니다.

아래 사진이 전체적인 구조 및 흐름입니다.

![image](https://github.com/Kim-YH-handong/GPU_Simple_loadbalancing_server/assets/82792998/a9e01c47-e899-46bc-b3ca-2bb27b1a3c61)


# Usage

1. git clone https://github.com/Kim-YH-handong/GPU_Simple_loadbalancing_server.git
2. AI Model Download Link: https://drive.google.com/file/d/1Vkg33jaWpl-sVpCLyz50h-qS5e1q38qZ/view?usp=sharing
3. 다운로드 받은 .pth파일을 아래와 같은 폴더 형식으로 압축해제 해주세요.


![image](https://github.com/Kim-YH-handong/GPU_Simple_loadbalancing_server/assets/82792998/02cfbd48-390c-4c04-a22f-f21ad8521bdc)

4. gpu server들을 각각 실행시켜주세요. [예시는 분리된 컴퓨터가 없어 Process를 분리해서 사용함]

![image](https://github.com/Kim-YH-handong/GPU_Simple_loadbalancing_server/assets/82792998/4aa3cff1-4bb7-4735-8d05-90bca68898cc)

5. main server를 실행시켜주세요. [4,5번 순서 꼭 지켜서 실행해주어야함]

![image](https://github.com/Kim-YH-handong/GPU_Simple_loadbalancing_server/assets/82792998/4a1fd7a8-b797-4193-ae94-11d52cda8a83)

6. 보낼 이미지를 폴더에 넣어서 아래와 같은 형식으로 위치시켜주세요.

![image](https://github.com/Kim-YH-handong/GPU_Simple_loadbalancing_server/assets/82792998/84656c01-18b2-43ae-a2cf-9faa0454b313)

7. 클라이언트를 실행시켜 보내고자하는 폴더 이름을 입력하면 됩니다.

![image](https://github.com/Kim-YH-handong/GPU_Simple_loadbalancing_server/assets/82792998/4bbf7e64-9417-4c14-97b2-1f620a9b6313)

# Demo

https://youtu.be/BTbf-hTBaSU
