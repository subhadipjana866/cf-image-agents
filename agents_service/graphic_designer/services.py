from openai import OpenAI
from .config import OPENAI_API_KEY, STABILITY_API_KEY, FLUX_API_KEY, GEMINI_API_KEY

# Bound for the raw OpenAI image-gen/edit clients below. Without an explicit
# timeout, a stalled connection hangs the calling thread (and whatever awaits
# it -- e.g. suggestionExecutors' ThreadPoolExecutor.result()) indefinitely
# with no exception raised, which looks like the request is just stuck.
IMAGE_API_TIMEOUT_SECONDS = 90
from .utils import makeApiRequest, profile, invoke_with_fallback, calculateCost, parseJson, apiLog
import time
import json
import base64
from io import BytesIO
from PIL import Image

def _geminiImageUsage(response_data: dict) -> dict:
    """Real (tokens_in, tokens_out) kwargs from a raw generateContent response's
    usageMetadata, for apiLog's actual_cost pricing. Gemini image calls report
    this on every response -- ignoring it is what used to force image cost
    onto a flat per-call guess instead of what the provider actually billed."""
    usage = response_data.get("usageMetadata", {}) or {}
    return {"tokens_in": usage.get("promptTokenCount"),
            "tokens_out": usage.get("candidatesTokenCount")}

# Functions for image generation using different APIs

@profile
def dalleGen(prompt, model,n):
    """Generate images using OpenAI's DALL-E API.

    Creates an OpenAI client, selects the appropriate size based on the
    model variant (``dall-e-2`` or ``dall-e-3``), and generates ``n``
    images.  Returns the response data containing base-64 encoded images.

    :param prompt: The text prompt for image generation.
    :type prompt: str
    :param model: The DALL-E model identifier (e.g. ``'dall-e-2 1024x1024'``).
    :type model: str
    :param n: Number of images to generate.
    :type n: int
    :returns: The raw API response data list containing base-64 image strings.
    :rtype: list
    :raises Exception: If the API call fails.
    """
    apiLog("openai", "dall-e", "dalleGen", status="start", cost=0.0)
    # print(chat_history)
    # print('check 1')
    try:
        client = OpenAI()
        if model.split(' ')[0] == 'dall-e-2':
            response = client.images.generate(
                model=model.split(' ')[0],
                prompt=prompt,
                # quality='hd' if model.split(' ')[1] == 'HD' else 'standard',
                n=n,
                response_format='b64_json'
            )
        else:
            response = client.images.generate(
                model=model.split(' ')[0],
                prompt=prompt,
                quality='hd' if model.split(' ')[1] == 'HD' else 'standard',
                n=n,
                response_format='b64_json'
            )
        # print(response)
        # print("check 2")
        if n == 1:
            return response.data[0].b64_json
        elif n > 1:
            return response.data
    except Exception as e:
        raise
@profile
def stableDiffGen(prompt, model, aspectRatio, references = None, negative_prompt = None):
    """Generate images using the Stability AI (Stable Diffusion) API.

    Sends a text prompt (and optional reference images) to the Stability AI
    ``stable-image/generate/sd3`` endpoint.  Supports reference-image
    conditioning via ``image`` and ``style_preset`` parameters.

    :param prompt: The text prompt for image generation.
    :type prompt: str
    :param model: The Stable Diffusion model identifier.
    :type model: str
    :param aspectRatio: The target aspect ratio (e.g. ``'16:9'``).
    :type aspectRatio: str
    :param references: Optional list of base-64 data URI reference images.
    :type references: list[str] | None
    :param negative_prompt: Optional negative prompt (unsupported on the
        ``ultra`` endpoint, so it's only applied for sd3/core models).
    :type negative_prompt: str | None
    :returns: The generated image as a base-64 encoded string.
    :rtype: str
    :raises Exception: If the API call fails or returns a non-200 status.
    """
    apiLog("stability", "stable-image", "stableDiffGen", status="start", cost=0.0)
    url = None
    is_ultra = 'ultra' in model
    if is_ultra:
        url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
    else:
        url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    modelN = None
    if 'large' in model:
        modelN = 'sd3.5-large-turbo'
    elif 'medium' in model:
        modelN = 'sd3.5-medium'
    if modelN:
        data = {
            "model": modelN,
            "prompt": prompt,
            "aspect_ratio": aspectRatio,
            "output_format": "webp",
        }
        if references:
            data['image'] = references[0].split(',')[1]
            data['mode'] = 'image-to-image'
            data['strength'] = 0.6
    else:
        data = {
            "prompt": prompt,
            "aspect_ratio": aspectRatio,
            "output_format": "webp",
        }
        if references:
            data['image'] = references[0].split(',')[1]
            data['strength'] = 0.6

    if negative_prompt and not is_ultra:
        data['negative_prompt'] = negative_prompt


    try:
        response = makeApiRequest(
            'POST',
            url,
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
                "accept": "application/json*"
            },
            files={"none": ''},
            data=data
        )
        if response.status_code == 200:
            # convert arraybuffer to base64
            data = response.json()
            return data['image']
        else:
            raise Exception(str(response.json()))
    except Exception as e:
        raise
@profile
def fluxGen(prompt, model, references = None):
    """Generate images using the Flux Pro API.

    Submits a generation request to the Flux ``flux-pro-1.1`` endpoint,
    then polls the ``get_result`` endpoint until the image is ready.
    Supports optional reference-image conditioning.

    :param prompt: The text prompt for image generation.
    :type prompt: str
    :param model: The Flux model identifier.
    :type model: str
    :param references: Optional list of base-64 data URI reference images.
    :type references: list[str] | None
    :returns: The generated image as a base-64 encoded string.
    :rtype: str
    :raises Exception: If the API call fails, times out, or is moderated.
    """
    apiLog("bfl", "flux", "fluxGen", status="start", cost=0.0)
    url = f"https://api.us.bfl.ai/v1/{model}"
    prompt = prompt
    data = {
        "prompt": prompt,
        "height": 1024,
        "output_format": "webp",
    }
    headers={
    'accept': 'application/json',
    'x-key': FLUX_API_KEY,
    'Content-Type': 'application/json',
    }

    if references:
        for i in range(1,len(references)):
            data[f'input_image_{i+1}'] = references[i].split(',')[1]
    
    try:
        response = makeApiRequest('POST', url,json=data, headers=headers)
        id = response.json()['id']
        url = "https://api.us.bfl.ai/v1/get_result"
        params = {
            "id": id
        }
        headers={
        'accept': 'application/json',
        'x-key': FLUX_API_KEY,
        }

        start_time = time.time()
        while True:

            if time.time() - start_time > 30:
                raise Exception("Timeout: Image generation took too long.")
            else:
                response = makeApiRequest('GET', url, params=params, headers=headers)
                res = response.json()
                if res['status'] == "Ready":
                    # Fetch the image over HTTP
                    response = makeApiRequest('GET', res['result']['sample'])
                    # Convert the raw bytes to a base64 string
                    base64_image = base64.b64encode(response.content).decode('utf-8')
                    return base64_image
                if res['status'] == 'Pending':
                    time.sleep(5)
    
                if res['status'] == 'Task not found' or res['status'] == 'Request Moderated' or res['status'] == 'Error' or res['status'] == 'Content moderated':
                    raise Exception("Error in generating image", res)
    except Exception as e:
        raise


def _build_image_gen_parts(prompt, references):
    """Assemble the Gemini ``parts`` array for an image-generation call.

    Each item in ``references`` may be either:

    * a plain base-64 data-URI string -- attached as an anonymous image
      (legacy behavior, unchanged); or
    * a ``{"label": <str>, "data": <data-uri>}`` dict -- the label is emitted
      as a text part immediately BEFORE its image so the model can bind that
      reference image to its role (product hero vs brand logo, etc.).

    This per-image labeling mirrors what the standalone SDK pipeline already
    does in ``image_generation.py`` and stops nano-banana from guessing which
    supplied image is the product and which is the logo.
    """
    parts = [{"text": prompt}]
    for ref in references or []:
        if isinstance(ref, dict):
            label = ref.get("label")
            data_uri = ref.get("data") or ""
            if label:
                parts.append({"text": label})
        else:
            data_uri = ref
        if not data_uri:
            continue
        parts.append({"inline_data": {"mime_type": data_uri.split(':')[1].split(';')[0], "data": data_uri.split(',')[1]}})
    return parts


@profile
def bananaImageGen(prompt, model, aspectRatio, references = None):
    """Generate images using the Nano-Banana (Gemini 2.5 Flash) API.

    Sends a text prompt (and optional reference images) to the
    ``gemini-2.5-flash-image`` Google Generative AI endpoint with
    ``responseModalities: ['IMAGE']``.

    :param prompt: The text prompt for image generation.
    :type prompt: str
    :param model: The model identifier.
    :type model: str
    :param aspectRatio: The target aspect ratio (e.g. ``'16:9'``).
    :type aspectRatio: str
    :param references: Optional list of base-64 data URI reference images.
    :type references: list[str] | None
    :returns: The generated image as a base-64 encoded string.
    :rtype: str
    :raises Exception: If the API call fails or returns a non-200 status.
    """
    apiLog("google", "nano-banana", "bananaImageGen", detail="gemini-2.5-flash-image-preview", status="start", cost=0.0)
    parts = _build_image_gen_parts(prompt, references)
    payload = {
        "contents": [{
        "parts": parts
    }],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
              "aspectRatio": aspectRatio
            }
        }
    }
    # API URL
    # `gemini-2.5-flash-image-preview` is not a served model id — it 404s on
    # every call. Verified against the live models list: the id is
    # `gemini-2.5-flash-image`. This is the last rung of the generation
    # fallback chain, so it 404'd exactly when the other two were failing.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key={GEMINI_API_KEY}"
    
    try:
        response = makeApiRequest(
            'POST',
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            response_data = response.json()
            # print('response json', response_data)
            # with open('data.json', 'w') as f:
            #     json.dump(response_data, f, indent=4)
            base64_data = response_data["candidates"][0]["content"]["parts"][-1]["inlineData"]["data"]
            apiLog("google", "nano-banana", "bananaImageGen", status="ok", **_geminiImageUsage(response_data))
            return base64_data
        else:
            raise Exception(f"Request failed with status {response.status_code}:", response.text)
    except Exception as e:
        raise
    
@profile
def bananaProImageGen(prompt, model, aspectRatio, references = None):
    """Generate images using the Nano-Banana Pro (Gemini 3 Pro) API.

    Sends a text prompt (and optional reference images) to the
    ``gemini-3-pro-image-preview`` Google Generative AI endpoint with
    ``responseModalities: ['IMAGE']``.

    :param prompt: The text prompt for image generation.
    :type prompt: str
    :param model: The model identifier.
    :type model: str
    :param aspectRatio: The target aspect ratio (e.g. ``'16:9'``).
    :type aspectRatio: str
    :param references: Optional list of base-64 data URI reference images.
    :type references: list[str] | None
    :returns: The generated image as a base-64 encoded string.
    :rtype: str
    :raises Exception: If the API call fails or returns a non-200 status.
    """
    apiLog("google", "nano-banana-pro", "bananaProImageGen", detail="gemini-3-pro-image-preview", status="start", cost=0.0)
    parts = _build_image_gen_parts(prompt, references)
    payload = {
        "contents": [{
        "parts": parts
    }],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
              "aspectRatio": aspectRatio
            }
        }
    }
    # API URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key={GEMINI_API_KEY}"
    
    try:
        response = makeApiRequest(
            'POST',
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            response_data = response.json()
            # print('response json', response_data)
            # with open('data.json', 'w') as f:
            #     json.dump(response_data, f, indent=4)
            base64_data = response_data["candidates"][0]["content"]["parts"][-1]["inlineData"]["data"]
            apiLog("google", "nano-banana-pro", "bananaProImageGen", status="ok", **_geminiImageUsage(response_data))
            return base64_data
        else:
            raise Exception(f"Request failed with status {response.status_code}:", response.text)
    except Exception as e:
        raise

@profile
def nanoBanana2ImageGen(prompt, model, aspectRatio, references = None):
    """Generate images using the Nano-Banana 2 (Gemini 3.1 Flash) API.

    Sends a text prompt (and optional reference images) to the
    ``gemini-3.1-flash-image-preview`` Google Generative AI endpoint with
    ``responseModalities: ['IMAGE']``.

    :param prompt: The text prompt for image generation.
    :type prompt: str
    :param model: The model identifier.
    :type model: str
    :param aspectRatio: The target aspect ratio (e.g. ``'16:9'``).
    :type aspectRatio: str
    :param references: Optional list of base-64 data URI reference images.
    :type references: list[str] | None
    :returns: The generated image as a base-64 encoded string.
    :rtype: str
    :raises Exception: If the API call fails or returns a non-200 status.
    """
    apiLog("google", "nano-banana-2", "nanoBanana2ImageGen", detail="gemini-3.1-flash-image-preview", status="start", cost=0.0)
    parts = _build_image_gen_parts(prompt, references)
    payload = {
        "contents": [{
        "parts": parts
    }],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
              "aspectRatio": aspectRatio
            }
        }
    }
    # API URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent?key={GEMINI_API_KEY}"

    try:
        response = makeApiRequest(
            'POST',
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            response_data = response.json()
            base64_data = response_data["candidates"][0]["content"]["parts"][-1]["inlineData"]["data"]
            apiLog("google", "nano-banana-2", "nanoBanana2ImageGen", status="ok", **_geminiImageUsage(response_data))
            return base64_data
        else:
            raise Exception(f"Request failed with status {response.status_code}:", response.text)
    except Exception as e:
        raise

def _data_uri_to_openai_file(data_uri: str, index: int):
    """Decode a ``data:<mime>;base64,<...>`` string into the
    ``(filename, bytes, mime)`` tuple form OpenAI's SDK accepts for
    ``images.edit``'s ``image=[...]`` list."""
    header, _, b64data = data_uri.partition(",")
    mime = header.split(";")[0].removeprefix("data:") or "image/png"
    ext = mime.split("/")[-1] or "png"
    return (f"reference_{index}.{ext}", base64.b64decode(b64data), mime)


@profile
def gptImageGen(prompt, model, aspectRatio, n, references=None):
    """Generate images using OpenAI's image generation API.

    Sends a text prompt to ``model`` via OpenAI's images endpoint. When
    ``references`` are supplied (e.g. an uploaded product photo or brand
    logo), uses the image-conditioned ``images.edit`` endpoint instead of
    plain text-to-image ``images.generate``, so those references actually
    influence the output instead of being silently dropped.  Returns a list
    of base-64 encoded image strings.

    :param prompt: The text prompt for image generation.
    :type prompt: str
    :param model: The model identifier, e.g. ``'gpt-image-2.5-flare'`` — resolved by
        the caller from the ``image`` role (see
        :func:`graphic_designer.nodes._gptImageApiModel`), never hardcoded here.
    :type model: str
    :param aspectRatio: The target aspect ratio, e.g. ``'3:4'``. Mapped to the
        closest of gpt-image's 3 native canvas sizes (see ``size_map`` below);
        exact matches (``1:1``, ``2:3``, ``3:2``) use the native size directly.
    :type aspectRatio: str
    :param n: Number of images to generate.
    :type n: int
    :param references: Optional list of ``data:<mime>;base64,<...>`` reference
        images (product photo, brand logo, templates, ...) to condition on.
    :type references: list[str] | None
    :returns: A list of base-64 encoded image strings.
    :rtype: list[str]
    :raises Exception: If the API call fails.
    """
    apiLog("openai", model, "gptImageGen", status="start", cost=0.0)
    try:
        size_map = {
            "1:1": "1024x1024",
            "2:3": "1024x1536",
            "3:2": "1536x1024",
        }
        size = size_map.get(aspectRatio)
        if size is None:
            # gpt-image only has 3 native canvases (square / 2:3 portrait / 3:2
            # landscape) -- unmapped ratios (3:4, 9:16, 16:9, ...) used to fall
            # back to square, so the model composed edge-to-edge for a 1:1
            # canvas and the later center-crop in _enforce_target_size had to
            # cut away up to a quarter of the image, slicing through content.
            # Route by orientation to the closest native canvas instead, so
            # that crop only trims a thin margin.
            try:
                w, h = (float(x) for x in aspectRatio.split(":"))
                ratio = w / h
            except (ValueError, AttributeError, ZeroDivisionError):
                ratio = 1.0
            if ratio > 1.05:
                size = "1536x1024"
            elif ratio < 0.95:
                size = "1024x1536"
            else:
                size = "1024x1024"
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=IMAGE_API_TIMEOUT_SECONDS)
        image_files = [
            _data_uri_to_openai_file(ref, i)
            for i, ref in enumerate(references or [])
            if isinstance(ref, str) and ref.startswith("data:")
        ]
        if image_files:
            result = client.images.edit(
                model=model,
                image=image_files,
                prompt=prompt,
                n=n,
                size=size,
                output_format='webp',
                quality='medium'
            )
        else:
            result = client.images.generate(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                output_format='webp',
                quality='medium'
            )
        usage = getattr(result, "usage", None)
        apiLog("openai", model, "gptImageGen", status="ok",
               tokens_in=getattr(usage, "input_tokens", None),
               tokens_out=getattr(usage, "output_tokens", None))
        return [img.b64_json for img in result.data]
    except Exception as e:
        # Was `return None`, which turned every provider error into a
        # "'NoneType' object is not subscriptable" TypeError one frame up --
        # the real message (e.g. 400 "The model 'gpt-image' does not exist")
        # never reached the logs or the fallback record. Both callers already
        # wrap this in try/except, so raising is safe.
        apiLog("openai", model, "gptImageGen", status="failed", detail=str(e))
        raise

@profile
def upscaleImage(imageBase64, scale: float = 4):
    """Upscale an image using the Stability AI Fast Upscaler API.

    Sends a base-64 encoded image to the ``stable-image/upscale/fast``
    Stability AI endpoint.  Polls for the result if the initial response
    is ``202 Accepted``.

    :param imageBase64: The input image as a base-64 encoded string
        (without the data-URI prefix).
    :type imageBase64: str
    :param scale: The upscale factor (default ``4``).
    :type scale: float
    :returns: The upscaled image as a base-64 encoded string.
    :rtype: str
    :raises Exception: If the API call fails or the upscale times out.
    """
    apiLog("stability", "upscale-fast", "upscaleImage", status="start", cost=0.0)

    # Decode once to be used by both methods
    try:
        decoded_bytes = base64.b64decode(imageBase64)
    except Exception as e:
        return f"Invalid Base64 input: {e}"

    # --- ATTEMPT 1: Stability AI API ---
    try:
        file_like_obj = BytesIO(decoded_bytes)

        response = makeApiRequest(
            'POST',
            'https://api.stability.ai/v2beta/stable-image/upscale/fast',
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
                "accept": "application/json*"
            },
            files={"image": file_like_obj},
            data={
                "output_format": "webp",
            }
        )

        # Check for HTTP errors (if makeApiRequest doesn't auto-raise)
        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")

        responseData = response.json()

        # Validation check
        if 'image' not in responseData:
            raise Exception("API response did not contain image data")

        apiLog("stability", "upscale-fast", "upscaleImage", status="ok")
        return responseData['image']

    except Exception as e:
        # We catch the error, log it, and allow the code to "fall through" to the next section
        apiLog("stability", "upscale-fast", "upscaleImage", status="failed", detail=str(e))


    # --- ATTEMPT 2: Local Lanczos (Fallback) ---
    try:
        input_image = BytesIO(decoded_bytes)
        img = Image.open(input_image)

        # Ensure scale is defined in your scope, otherwise set a default
        current_scale = scale if 'scale' in locals() else 2 

        new_size = (
            int(img.width * current_scale),
            int(img.height * current_scale)
        )

        # High-quality downsampling/upscaling filter
        upscaled = img.resize(new_size, Image.LANCZOS)

        buffered = BytesIO()
        upscaled.save(buffered, format="WEBP")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Local Lanczos resize — no provider was called, so this is free;
        # logged as its own zero-cost call so it doesn't read as a missing
        # Stability charge in api_calls.log.
        apiLog("local", "lanczos-resize", "upscaleImage", status="ok", cost=0.0)
        return img_str

    except Exception as e:
        return str(e)


@profile
def sugBananaEditor(base64image, prompt):
    """Edit an image using the Gemini 3.1 Flash Image API for suggestion execution.

    Sends a base-64 encoded image and an editing prompt to the
    ``gemini-3.1-flash-image-preview`` Google Generative AI endpoint.  Returns the
    edited image as a base-64 string.  Used specifically by the suggestion
    executor pipeline.

    :param base64image: The input image as a base-64 encoded string
        (without the data-URI prefix).
    :type base64image: str
    :param prompt: The editing instruction / nano-prompt.
    :type prompt: str
    :returns: The edited image as a base-64 encoded string.
    :rtype: str
    :raises Exception: If the API call fails or returns a non-200 status.
    """
    # This hits the same gemini-3.1-flash-image-preview endpoint as
    # nanoBanana2ImageGen (see below), so it bills/logs as "nano-banana-2" —
    # not "nano-banana" ($0.06 for gemini-2.5-flash-image, a different model).
    apiLog("google", "nano-banana-2", "sugBananaEditor", detail="gemini-3.1-flash-image-preview", status="start", cost=0.0)
    try:
        if not GEMINI_API_KEY:
            raise Exception("GEMINI_API_KEY not set")
        data_url = base64image

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": 'image/webp',
                                     "data": data_url}}
                ]
            }],
            "generationConfig": {"responseModalities": ["IMAGE"]}
        }
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-3.1-flash-image-preview:generateContent"
               f"?key={GEMINI_API_KEY}")
        
        response = makeApiRequest('POST', url, 
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}:", response.text)
        response_data = response.json()
        # save the response as a json file for debugging
        # with open("banana_response.json", "w") as f:
        #     json.dump(response_data, f, indent=4)

        base64_data = response_data["candidates"][0]["content"]["parts"][-1]["inlineData"]["data"]
        apiLog("google", "nano-banana-2", "sugBananaEditor", status="ok", **_geminiImageUsage(response_data))
        return base64_data

    except Exception as e:
        raise


@profile
def _openai_edit_size(aspect_ratio):
    """Nearest OpenAI image canvas for ``aspect_ratio``.

    Same mapping gptImageGen uses. This exists because the edit endpoint has no
    "keep the input's shape" option -- omitting ``size`` defaults to square, which
    silently squashes every non-square design. Callers MUST pass the real ratio.
    """
    try:
        w, h = (float(x) for x in (aspect_ratio or "1:1").split(":"))
        ratio = w / h
    except (ValueError, AttributeError, ZeroDivisionError, TypeError):
        ratio = 1.0
    if ratio > 1.05:
        return "1536x1024"
    if ratio < 0.95:
        return "1024x1536"
    return "1024x1024"


def sugGptImageEditor(base64image, prompt, model="gpt-image-2.5-flare", aspect_ratio="1:1"):
    """Edit an image with an OpenAI image model, for suggestion execution.

    Primary editor for the suggestion path (``gpt-image-2.5-flare``); also serves as the
    fallback editor when :func:`sugBananaEditor` is tried first. Sends the base-64
    image plus the compiled nano-prompt to ``images/edits``.

    ``aspect_ratio`` is REQUIRED in practice: this used to hardcode
    ``size="1024x1024"``, which was tolerable while this was a rarely-hit fallback
    but squashes every portrait/landscape design once it becomes the primary path.
    Callers must also check ``_gpt_image_2_supports_ratio`` first -- gpt-image-2.5-flare has
    only 3 native canvases and distorts anything meaningfully off them.

    :param base64image: The input image as a base-64 encoded string
        (without the data-URI prefix).
    :type base64image: str
    :param prompt: The editing instruction / nano-prompt.
    :type prompt: str
    :param model: OpenAI image model id to send.
    :type model: str
    :param aspect_ratio: Source ratio like ``'2:3'``, mapped to the nearest canvas.
    :type aspect_ratio: str
    :returns: The edited image as a base-64 encoded string.
    :rtype: str
    :raises Exception: If the API call fails.
    """
    size = _openai_edit_size(aspect_ratio)
    apiLog("openai", model, "sugGptImageEditor", status="start", cost=0.0,
           detail=f"size={size} aspect={aspect_ratio}")
    image_bytes = base64.b64decode(base64image)
    image_file = BytesIO(image_bytes)
    image_file.name = "image.webp"

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=IMAGE_API_TIMEOUT_SECONDS)
    result = client.images.edit(
        model=model,
        image=image_file,
        prompt=prompt,
        n=1,
        size=size,
        output_format='webp',
        quality='medium',
    )
    usage = getattr(result, "usage", None)
    apiLog("openai", model, "sugGptImageEditor", status="ok",
           tokens_in=getattr(usage, "input_tokens", None),
           tokens_out=getattr(usage, "output_tokens", None))
    return result.data[0].b64_json
