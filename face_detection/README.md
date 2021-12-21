# Поиск ключевых точек на лице
В программе реализована сверточная нейронная сеть на фреймворке pytorch-lightning, выполняющая регрессию 14 ключевых точек лица.
![example_collage.png](https://github.com/dmitrylala/cv_msu/blob/main/face_detection/example_collage.png)

## Inference
Для корректного инференса лучше подавать изображения лиц, не содержащие верхней части головы, поскольку нейросеть обучалась на схожих данных.
Запуск можно осуществить через командную строку:
~~~
python main.py (model.ckpt) (img_src_dir) (output_dir)
~~~
