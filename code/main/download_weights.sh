
FILE=result/Top_1024x768_COTTON/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19_tXE7VSoT_z2JRZXAKuObM8T_w39WLS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19_tXE7VSoT_z2JRZXAKuObM8T_w39WLS" -O result/Top_1024x768_COTTON/weights.zip && rm -rf /tmp/cookies.txt
    unzip result/Top_1024x768_COTTON/weights.zip -d result/Top_1024x768_COTTON/
fi

FILE=result/Top_1024x768_DressCode/weights
if ! test -d "$FILE"; then
    echo "$FILE not exists."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HPg4Qv3vorp7D9Cd4rn2inKx8J-nQJI4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HPg4Qv3vorp7D9Cd4rn2inKx8J-nQJI4" -O result/Top_1024x768_DressCode/weights.zip && rm -rf /tmp/cookies.txt
    unzip result/Top_1024x768_DressCode/weights.zip -d result/Top_1024x768_DressCode/
fi

