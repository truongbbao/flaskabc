from flask import Flask
from flask_cors import CORS, cross_origin #CORS là một cơ chế cho phép nhiều tài nguyên khác nhau của một trang web có thể được truy vấn từ domain
from flask import request,jsonify
from ultralyticsplus import YOLO, postprocess_classify_output
import os
import cv2

modelv8 = YOLO('best.pt')
modelv8ct = YOLO('ctbest.pt')
# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"

@app.route('/ctscan', methods=['POST'] )
def predictct():
        # đọc ảnh
        image = request.files['img']

        if image:
            # Lưu file
            path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            print("Save = ", path_to_save)
            image.save(path_to_save)
            frame = cv2.imread(path_to_save)

            # Nhận diên qua model Yolov6
            results = modelv8ct.predict(frame)
            processed_result = postprocess_classify_output(modelv8ct, result=results[0])
            # Xóa ảnh lưu tạm
            try:
                os.remove(path_to_save)
            except OSError as e:
                return "Error: %s - %s." % (e.filename, e.strerror)
            del frame
            return jsonify(**processed_result)

        return

@app.route('/pneu', methods=['POST'] )
def predict():
        # đọc ảnh
        image = request.files['img']

        if image:
            # Lưu file
            path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            print("Save = ", path_to_save)
            image.save(path_to_save)
            frame = cv2.imread(path_to_save)

            # Nhận diên qua model Yolov6
            results = modelv8.predict(frame)
            processed_result = postprocess_classify_output(modelv8, result=results[0])
            # Trả về đường dẫn tới file ảnh đã bounding box
            try:
                os.remove(path_to_save)
            except OSError as e:
                return "Error: %s - %s." % (e.filename, e.strerror)
            del frame
            return jsonify(**processed_result)

        return


# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')