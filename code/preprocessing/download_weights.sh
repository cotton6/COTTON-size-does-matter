
FILE=CIHP_PARSING/checkpoint
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QFUujOeUY9YRz5_Mq-TBYNfP4ft-yhMJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QFUujOeUY9YRz5_Mq-TBYNfP4ft-yhMJ" -O CIHP_PARSING/checkpoint.zip && rm -rf /tmp/cookies.txt
    unzip CIHP_PARSING/checkpoint.zip -d CIHP_PARSING
fi

FILE=U2Net/saved_models
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Mxf0T_IIZfN6DG5k7Ibrh2GvwZ11N7u5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Mxf0T_IIZfN6DG5k7Ibrh2GvwZ11N7u5" -O U2Net/saved_models.zip && rm -rf /tmp/cookies.txt
    unzip U2Net/saved_models.zip -d U2Net
fi

FILE=Sleeve_Classifier/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BYMPYjhzURv6p_Ow0LWWJUnpgZnNHCby' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BYMPYjhzURv6p_Ow0LWWJUnpgZnNHCby" -O Sleeve_Classifier/weights.zip && rm -rf /tmp/cookies.txt
    unzip Sleeve_Classifier/weights.zip -d Sleeve_Classifier
fi

FILE=lower_clf/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WjBsVuVH3A3ZkDHiDkP4eGg5s9t8tCjY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WjBsVuVH3A3ZkDHiDkP4eGg5s9t8tCjY" -O lower_clf/weights.zip && rm -rf /tmp/cookies.txt
    unzip lower_clf/weights.zip -d lower_clf
fi

FILE=Cloth2Skeleton/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1M5YE9HhG73x4EAcBu9WP-31OR2Awyh6W' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1M5YE9HhG73x4EAcBu9WP-31OR2Awyh6W" -O Cloth2Skeleton/weights.zip && rm -rf /tmp/cookies.txt
    unzip Cloth2Skeleton/weights.zip -d Cloth2Skeleton
fi

FILE=ClothSegmentation/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1j11XCO6cD3nYo-4ObQHuYwe7Ocy5BoGw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1j11XCO6cD3nYo-4ObQHuYwe7Ocy5BoGw" -O ClothSegmentation/weights.zip && rm -rf /tmp/cookies.txt
    unzip ClothSegmentation/weights.zip -d ClothSegmentation
fi

FILE=Self-Correction-Human-Parsing/exp-schp-201908301523-atr.pth
if ! test -f "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1airhKX-o8AIxs3M0uFU4BNvwc5YZo7Xd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1airhKX-o8AIxs3M0uFU4BNvwc5YZo7Xd" -O Self-Correction-Human-Parsing/exp-schp-201908301523-atr.pth && rm -rf /tmp/cookies.txt
fi

