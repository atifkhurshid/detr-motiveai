import json


def write_annotations(targets, predictions, categories, filepath):

    assert len(targets) == len(predictions)

    i = 0
    images = []
    annotations = []

    for target, prediction in zip(targets, predictions):

        image = {
            'file_name': target['file_name'],
            'id': target['id'],
            'height': target['height'],
            'width': target['width'],
        }
        images.append(image)

        for bbox, label in zip(prediction['boxes'], prediction['labels']):
            ann = {
                'image_id': image['id'],
                'bbox': bbox,
                'category_id': label,
                'id': i,
                'iscrowd': 0,
                'area': bbox[2] * bbox[3],
            }
            annotations.append(ann)
            i += 1
    
    data = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    __jwrite(filepath, data)


def __jwrite(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
    