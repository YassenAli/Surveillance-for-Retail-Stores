{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.12","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"nvidiaTeslaT4","dataSources":[{"sourceId":89993,"databundleVersionId":10699058,"sourceType":"competition"}],"dockerImageVersionId":30918,"isInternetEnabled":true,"language":"python","sourceType":"notebook","isGpuEnabled":true}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"pip install deepface","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-03-05T19:02:58.317829Z","iopub.execute_input":"2025-03-05T19:02:58.318069Z","iopub.status.idle":"2025-03-05T19:03:01.994517Z","shell.execute_reply.started":"2025-03-05T19:02:58.318048Z","shell.execute_reply":"2025-03-05T19:03:01.993256Z"}},"outputs":[{"name":"stdout","text":"Requirement already satisfied: deepface in /usr/local/lib/python3.10/dist-packages (0.0.93)\nRequirement already satisfied: requests>=2.27.1 in /usr/local/lib/python3.10/dist-packages (from deepface) (2.32.3)\nRequirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (1.26.4)\nRequirement already satisfied: pandas>=0.23.4 in /usr/local/lib/python3.10/dist-packages (from deepface) (2.2.3)\nRequirement already satisfied: gdown>=3.10.1 in /usr/local/lib/python3.10/dist-packages (from deepface) (5.2.0)\nRequirement already satisfied: tqdm>=4.30.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (4.67.1)\nRequirement already satisfied: Pillow>=5.2.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (11.0.0)\nRequirement already satisfied: opencv-python>=4.5.5.64 in /usr/local/lib/python3.10/dist-packages (from deepface) (4.10.0.84)\nRequirement already satisfied: tensorflow>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (2.17.1)\nRequirement already satisfied: keras>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (3.5.0)\nRequirement already satisfied: Flask>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from deepface) (3.1.0)\nRequirement already satisfied: flask-cors>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from deepface) (5.0.1)\nRequirement already satisfied: mtcnn>=0.1.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (1.0.0)\nRequirement already satisfied: retina-face>=0.0.1 in /usr/local/lib/python3.10/dist-packages (from deepface) (0.0.17)\nRequirement already satisfied: fire>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (0.7.0)\nRequirement already satisfied: gunicorn>=20.1.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (23.0.0)\nRequirement already satisfied: termcolor in /usr/local/lib/python3.10/dist-packages (from fire>=0.4.0->deepface) (2.5.0)\nRequirement already satisfied: Werkzeug>=3.1 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (3.1.3)\nRequirement already satisfied: Jinja2>=3.1.2 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (3.1.4)\nRequirement already satisfied: itsdangerous>=2.2 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (2.2.0)\nRequirement already satisfied: click>=8.1.3 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (8.1.7)\nRequirement already satisfied: blinker>=1.9 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (1.9.0)\nRequirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from gdown>=3.10.1->deepface) (4.12.3)\nRequirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from gdown>=3.10.1->deepface) (3.17.0)\nRequirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from gunicorn>=20.1.0->deepface) (24.2)\nRequirement already satisfied: absl-py in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (1.4.0)\nRequirement already satisfied: rich in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (13.9.4)\nRequirement already satisfied: namex in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (0.0.8)\nRequirement already satisfied: h5py in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (3.12.1)\nRequirement already satisfied: optree in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (0.13.1)\nRequirement already satisfied: ml-dtypes in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (0.4.1)\nRequirement already satisfied: joblib>=1.4.2 in /usr/local/lib/python3.10/dist-packages (from mtcnn>=0.1.0->deepface) (1.4.2)\nRequirement already satisfied: lz4>=4.3.3 in /usr/local/lib/python3.10/dist-packages (from mtcnn>=0.1.0->deepface) (4.4.3)\nRequirement already satisfied: mkl_fft in /usr/local/lib/python3.10/dist-packages (from numpy>=1.14.0->deepface) (1.3.8)\nRequirement already satisfied: mkl_random in /usr/local/lib/python3.10/dist-packages (from numpy>=1.14.0->deepface) (1.2.4)\nRequirement already satisfied: mkl_umath in /usr/local/lib/python3.10/dist-packages (from numpy>=1.14.0->deepface) (0.1.1)\nRequirement already satisfied: mkl in /usr/local/lib/python3.10/dist-packages (from numpy>=1.14.0->deepface) (2025.0.1)\nRequirement already satisfied: tbb4py in /usr/local/lib/python3.10/dist-packages (from numpy>=1.14.0->deepface) (2022.0.0)\nRequirement already satisfied: mkl-service in /usr/local/lib/python3.10/dist-packages (from numpy>=1.14.0->deepface) (2.4.1)\nRequirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.23.4->deepface) (2.9.0.post0)\nRequirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.23.4->deepface) (2025.1)\nRequirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.23.4->deepface) (2025.1)\nRequirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->deepface) (3.4.1)\nRequirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->deepface) (3.10)\nRequirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->deepface) (2.3.0)\nRequirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->deepface) (2025.1.31)\nRequirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.6.3)\nRequirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (24.3.25)\nRequirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (0.6.0)\nRequirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (0.2.0)\nRequirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (18.1.1)\nRequirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (3.4.0)\nRequirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (3.20.3)\nRequirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (75.1.0)\nRequirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.17.0)\nRequirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (4.12.2)\nRequirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.17.0)\nRequirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.68.1)\nRequirement already satisfied: tensorboard<2.18,>=2.17 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (2.17.1)\nRequirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (0.37.1)\nRequirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow>=1.9.0->deepface) (0.45.1)\nRequirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from Jinja2>=3.1.2->Flask>=1.1.2->deepface) (3.0.2)\nRequirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow>=1.9.0->deepface) (3.7)\nRequirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow>=1.9.0->deepface) (0.7.2)\nRequirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->gdown>=3.10.1->deepface) (2.6)\nRequirement already satisfied: intel-openmp>=2024 in /usr/local/lib/python3.10/dist-packages (from mkl->numpy>=1.14.0->deepface) (2024.2.0)\nRequirement already satisfied: tbb==2022.* in /usr/local/lib/python3.10/dist-packages (from mkl->numpy>=1.14.0->deepface) (2022.0.0)\nRequirement already satisfied: tcmlib==1.* in /usr/local/lib/python3.10/dist-packages (from tbb==2022.*->mkl->numpy>=1.14.0->deepface) (1.2.0)\nRequirement already satisfied: intel-cmplr-lib-rt in /usr/local/lib/python3.10/dist-packages (from mkl_umath->numpy>=1.14.0->deepface) (2024.2.0)\nRequirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown>=3.10.1->deepface) (1.7.1)\nRequirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=2.2.0->deepface) (3.0.0)\nRequirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=2.2.0->deepface) (2.19.1)\nRequirement already satisfied: intel-cmplr-lib-ur==2024.2.0 in /usr/local/lib/python3.10/dist-packages (from intel-openmp>=2024->mkl->numpy>=1.14.0->deepface) (2024.2.0)\nRequirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=2.2.0->deepface) (0.1.2)\nNote: you may need to restart the kernel to use updated packages.\n","output_type":"stream"}],"execution_count":1},{"cell_type":"code","source":"import sys\nimport numpy as np\nimport os\nimport shutil\nsys.setrecursionlimit(10000)\n\nfrom deepface import DeepFace\n\nclass DeepFaceRecognizer:\n    def __init__(self, db_path, model_name=\"ArcFace\", distance_metric=\"cosine\", threshold=0.4, enforce_detection=False):\n        self.db_path = db_path\n        self.model_name = model_name\n        self.distance_metric = distance_metric\n        self.threshold = threshold\n        self.enforce_detection = enforce_detection\n\n    def recognize(self, test_img_path):\n        try:\n            # DeepFace.find will search the db_path for similar faces using the chosen model and metric.\n            results = DeepFace.find(\n                img_path=test_img_path,\n                db_path=self.db_path,\n                model_name=self.model_name,\n                distance_metric=self.distance_metric,\n                enforce_detection=self.enforce_detection\n            )\n            \n            if not results or results[0].empty:\n                return \"doesn't_exist\", None\n            \n            df = results[0]\n            best_match = df.iloc[0]\n            best_distance = best_match[\"distance\"]\n            if best_distance < self.threshold:\n                return best_match[\"identity\"], best_distance\n            else:\n                return \"doesn't_exist\", best_distance\n        except Exception as e:\n            print(\"Error in recognize:\", str(e))\n            return \"doesn't_exist\", None\n\n# --- Testing Code ---\nif __name__ == '__main__':\n    # Original read-only database path\n    original_db_path = \"/kaggle/input/surveillance-for-retail-stores/face_identification/face_identification/train\"\n    # Writable database path in /kaggle/working/\n    writable_db_path = \"/kaggle/working/face_identification_train\"\n    \n    # Copy the database to a writable directory if it doesn't already exist\n    if not os.path.exists(writable_db_path):\n        shutil.copytree(original_db_path, writable_db_path)\n        print(f\"Copied database from {original_db_path} to {writable_db_path}\")\n    else:\n        print(f\"Using existing database at {writable_db_path}\")\n\n    test_img_path = \"/kaggle/input/surveillance-for-retail-stores/face_identification/face_identification/test/200.jpg\"\n    \n    recognizer = DeepFaceRecognizer(\n        db_path=writable_db_path,  # Use the writable path\n        model_name=\"ArcFace\",\n        distance_metric=\"cosine\",\n        threshold=0.3,\n        enforce_detection=False\n    )\n    \n    identity, distance = recognizer.recognize(test_img_path)\n    print(\"Recognized Identity:\", identity)\n    print(\"Distance:\", distance)","metadata":{"_uuid":"bd569b83-61e7-4994-8f7d-a4904c8db37d","_cell_guid":"8cbeff8a-010d-4519-8c53-4edb758c3460","trusted":true,"collapsed":false,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-05T19:36:45.249542Z","iopub.execute_input":"2025-03-05T19:36:45.249822Z","iopub.status.idle":"2025-03-05T19:36:47.334341Z","shell.execute_reply.started":"2025-03-05T19:36:45.249801Z","shell.execute_reply":"2025-03-05T19:36:47.333403Z"}},"outputs":[{"name":"stdout","text":"Using existing database at /kaggle/working/face_identification_train\n25-03-05 19:36:46 - Searching /kaggle/input/surveillance-for-retail-stores/face_identification/face_identification/test/200.jpg in 6839 length datastore\n25-03-05 19:36:47 - find function duration 2.029614210128784 seconds\nRecognized Identity: /kaggle/working/face_identification_train/person_101/219.jpg\nDistance: 0.07634943119823356\n","output_type":"stream"}],"execution_count":7}]}