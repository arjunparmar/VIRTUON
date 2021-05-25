## Step to run this project:

> NOTE: To run this project you will need Graphics Card (GPU).
> To run on CPU please checkout here.

#### 1. Clone this repo or download the zip file.
#### 2. Ensure you have Python3 and pip already installed on your system.
#### 3. Navigate to VIRTUON/model_deployment
#### 4.1 Setup virtual environment in Ubuntu
 - Run the following commands in terminal
 ```
    pip3 install virtualenv 
    virtualenv venv
    source venv/bin/activate
 ```
 
#### 4.2 Setup virtual environment in Windows in CMD
 - Run the following commands in cmd.exe
 ```
    pip3 install virtualenv 
    virtualenv venv
    source C:\> venv\Scripts\activate.bat
 ```
 
 #### 4.3 Setup virtual environment in Windows in PowerShell
  - Run the following commands in PowerShell.exe
 ```
    pip3 install virtualenv 
    virtualenv venv
    source PS C:\> venv\Scripts\Activate.ps1
 ```
 For more detials about Virtual Enviroments in Python checkout [this.](https://docs.python.org/3/library/venv.html)
 
#### 5. Install the dependencies
 - Note you should be in model_deployment folder and ensure requirements.txt exist.
 - Run the following command
 ``` 
  pip3 install -r requirements.txt
 ```
 - Note: This will take a while. It will be of approx. 850MB

#### 6. Starting the server
 - When starting it for first time, run the following command.
```
  python3 manage.py migrate
```
 - Now run
```
  python3 manage.py runserver
```
 - Copy and paste this address in your browser
```
  http://127.0.0.1:8000/
``` 
 - Tryon :)
   
