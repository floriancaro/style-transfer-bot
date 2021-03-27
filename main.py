#!/usr/bin/env python3

import tweepy
import logging
from config import create_api
import time
from model_files.ml_model import *
import requests
import os
from PIL import Image
import io # to convert from Image to File object

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def check_mentions(api, keywords, since_id, retrieved_nr, image_url_list, posted_list):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    
    # get tweets with mentions since the last tweet (since_id) checked
    for tweet in tweepy.Cursor(api.mentions_timeline,
        since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)
        
        # ignore replies
        if tweet.in_reply_to_status_id is not None:
            continue

        # check if the keywords are in the tweet
        if any(keyword in tweet.text.lower() for keyword in keywords):
            # check if there is an image in the tweet
            for tweet in tweepy.Cursor(api.mentions_timeline,since_id=since_id).items():
                if("media" in tweet.entities):
                    media_url = tweet.entities["media"][0]["media_url_https"]
                    
                    im = Image.open(requests.get(media_url, stream=True).raw)
                    # im.save("retrieved/retrieved_image"+str(retrieved_nr)+".png")
                    retrieved_nr += 1

                    im = imageio.imread(media_url)

                    image_url_list.append(media_url)
                    logger.info(f"Answering to {tweet.user.name}")

                    # # follow user who posted the Tweet
                    # if not tweet.user.following:
                    #     tweet.user.follow()

                    # create the synthesized image through style transfer
                    synthesized_image = synthesize_image(im, num_iterations = 200)
                    
                    synthesized_image = prepare_image(synthesized_image)

                    image_to_post = io.BytesIO()
                    synthesized_image.save(image_to_post, "JPEG")
                    image_to_post.seek(0)

                    # post synthesized image
                    api.update_with_media("synthesized_image", file = image_to_post)
                    # api.update_with_media("retrieved/retrieved_image0.png") # for testing

                    # im.save('../posted/' + synthesized_image) 
                    print("Posted!")
                        
    return new_since_id

def main():
    api = create_api()
    since_id = 1
    nr = 0
    image_urls = []
    posted = []
    while True:
        since_id = check_mentions(api, ["content", "style"], since_id, nr, image_urls, posted)
        logger.info("Waiting...")
        time.sleep(60)

while True:
    main()

# if __name__ == "__main__":
#     main()

# # The run method starts our flask application service
# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=8080) # http://0.0.0.0 often doesn't work, you will likely need to switch to http://localhost:8080 in your browser.