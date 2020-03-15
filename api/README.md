# Starter for deploying [fast.ai](https://www.fast.ai) models on [Render](https://render.com)

This folder contains the API used to process the posts, apply the Deep Learning model and then return the results.

## How To Use The API

You can test your changes locally by installing Docker and using the following command:

```
docker build -t coruscant-api . && docker run --rm -it -p 5000:5000 coruscant-api
```

The guide for production deployment to Render is at https://course.fast.ai/deployment_render.html.

Please use [Render's fast.ai forum thread](https://forums.fast.ai/t/deployment-platform-render/33953) for questions and support.
