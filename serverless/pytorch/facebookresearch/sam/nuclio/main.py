# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
import imagehash
from PIL import Image
import io
import glob
from model_handler import ModelHandler

def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")

def get_cached(context, image):
    cache_dir = '/var/lib/sam-model/embeddings'
    img_hash = imagehash.phash(image)
    hits = glob.glob(f"{cache_dir}/{img_hash}_*")
    if len(hits) == 1:
        context.logger.info(f"cache hit {hits[0]}")
        return open(hits[0], 'rb').read()
    elif len(hits) > 1:
        context.logger.info(f"cache hash collision {img_hash}")
    else:
        context.logger.info(f"cache miss {img_hash}")

def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    features = get_cached(context, image)
    if not features:
        image = image.convert("RGB")  #  to make sure image comes in RGB
        features_raw = context.user_data.model.handle(image)
        features = features_raw.cpu().numpy() if features_raw.is_cuda else features_raw.numpy()

    return context.Response(body=json.dumps({
            'blob': base64.b64encode(features).decode(),
        }),
        headers={},
        content_type='application/json',
        status_code=200
    )
