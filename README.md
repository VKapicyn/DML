# Курсовой проект по Deep Learning 

Для запуска понадобиться [Docker](https://docker.com.).

## Запуск сервера
В папке с проектом: 
```sh
docker build -t dml_vkapicyn .
docker run -p 8000:8000 dml_vkapicyn
```
Приложение доступно по адресу http:localhost:8000/

## Повторное обучение

1. Скачайте датасет, распакуйте архив в /dataset
2. Переместите датасет в контейнер.
```sh
docker cp ВАШ_ПУТЬ_ДО_dataset container:/
```
3. Откройте терминал docker-образа
```sh
docker -it dml_train_vkapicyn /bin/bash
```
4. Запустите файл с обучением. --age для получение модели только с возрастом и --gender только с полом.
```sh
python trayn.py --all
```

Выполнил студент группы М06-006ск
Капицын Владислав