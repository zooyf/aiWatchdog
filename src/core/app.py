from fastapi import FastAPI, APIRouter
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html, get_swagger_ui_oauth2_redirect_html
from starlette.staticfiles import StaticFiles


def create_app(prefix, **kwargs):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=f"{prefix}/openapi.json", **kwargs)
    app.mount(f"{prefix}/static", StaticFiles(directory="assets"), name="static")
    api_router = APIRouter(prefix=prefix)

    @api_router.get("/")
    async def health_check():
        return {"status": "ok", "message": "服务运行正常"}

    # 自定义Swagger UI文档路由
    @api_router.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + "ChatMCP - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url=f"{prefix}/static/docs-openapi/swagger-ui-bundle.js",
            swagger_css_url=f"{prefix}/static/docs-openapi/swagger-ui.css"
        )

    @api_router.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
    async def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    @api_router.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=app.title + " - ReDoc",
            redoc_js_url=f"{prefix}/static/docs-openapi/redoc.standalone.js",
        )

    app.include_router(api_router)

    return app


app = create_app('')
