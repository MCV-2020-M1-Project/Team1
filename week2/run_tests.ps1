python fastsearch.py -r '../resources/BBDD' -q '../resources/qsd1_w2' --use_masks --d 'lab_histogram_3d_blocks' --metric 'hellinger' -b 24 -k 1
python fastsearch.py -r '../resources/BBDD' -q '../resources/qsd1_w2' --use_masks --d 'lab_histogram_3d_blocks' --metric 'hellinger' -b 24 -k 5
python fastsearch.py -r '../resources/BBDD' -q '../resources/qsd1_w2' --use_masks --d 'ycrcb_histogram_3d_blocks' --metric 'hellinger' -b 24 -k 1
python fastsearch.py -r '../resources/BBDD' -q '../resources/qsd1_w2' --use_masks --d 'ycrcb_histogram_3d_blocks' --metric 'hellinger' -b 24 -k 5

python fastsearch.py -r '../resources/BBDD' -q '../resources/qsd1_w2' --use_masks --d 'lab_histogram_3d_blocks' --metric 'js_div' -b 24 -k 1
python fastsearch.py -r '../resources/BBDD' -q '../resources/qsd1_w2' --use_masks --d 'lab_histogram_3d_blocks' --metric 'js_div' -b 24 -k 5
python fastsearch.py -r '../resources/BBDD' -q '../resources/qsd1_w2' --use_masks --d 'ycrcb_histogram_3d_blocks' --metric 'js_div' -b 24 -k 1
python fastsearch.py -r '../resources/BBDD' -q '../resources/qsd1_w2' --use_masks --d 'ycrcb_histogram_3d_blocks' --metric 'js_div' -b 24 -k 5
