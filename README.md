# 一键直拍

一键直拍是一个本地可运行的单人物直拍生成工具。

它面向这样的场景：用户手里有一段多人横屏视频，只想快速生成其中某一个人的专属直拍，不想自己逐帧打关键帧、拉裁剪框、反复导出。

当前版本已经支持两种成片模式：

- 生成竖屏直拍：输出 `9:16`，人物居中，镜头跟随并适度放大
- 生成横屏直拍：保持横屏比例，目标人物清晰，其余左右区域做柔和模糊弱化

## 功能概览

- 上传本地 `MP4 / MOV / MKV` 视频
- 通过时间轴裁出保留片段
- 自动从保留片段中挑选更适合选人的代表帧
- 自动检测画面中的人物候选，并支持用户点击目标人物
- 基于身份锚点做单人物跟踪
- 对低置信或异常帧做人工纠偏
- 生成竖屏直拍或横屏直拍低清预览
- 导出高清 `MP4`

## 项目截图

桌面端工作台：

![桌面端界面](./docs/screenshots/ui-desktop.png)

移动端界面：

![移动端界面](./docs/screenshots/ui-mobile.png)

横屏直拍效果示例：

![横屏直拍示例](./docs/screenshots/horizontal-preview.jpg)

## 产品流程

```text
上传视频
  -> 裁保留片段
  -> AI 选代表帧
  -> 用户点选目标人物
  -> 身份跟踪
  -> 风险帧复核 / 人工纠偏
  -> 选择竖屏或横屏直拍
  -> 生成预览
  -> 导出高清成片
```

## 快速开始

首次使用：

1. 双击 `安装AI能力.cmd`
2. 确保本机可以使用 `FFmpeg`
3. 双击 `启动一键直拍.cmd`
4. 浏览器会自动打开本地地址

英文脚本和中文脚本是同一套入口：

- `install-ai.cmd` / `安装AI能力.cmd`
- `start-onetake.cmd` / `启动一键直拍.cmd`

如果某台 Windows 电脑对中文脚本名兼容不好，直接使用英文脚本即可。

也可以手动启动：

```powershell
python server.py
```

## 安装排查

常见问题和处理方式见：

- [安装与启动排查](./docs/TROUBLESHOOTING.md)

## 目录说明

- `index.html`、`styles.css`、`app.js`：前端界面
- `server.py`：本地服务与主流程编排
- `sam2_backend.py`：高精度跟踪后端适配层
- `scripts/`：模型下载、启动和验证脚本
- `models/`：检测模型配置与标签
- `cloud/`：云端部署预留文件
- `docs/`：产品文档与截图
- `outputs/`：本地导出结果
- `work/`：调试与验证过程文件

## 文档

- [产品规格](./docs/PRODUCT_SPEC.md)
- [产品路线图](./docs/ROADMAP.md)
- [GitHub 上传说明](./docs/REPOSITORY_UPLOAD_GUIDE.md)
- [GitHub 上传前最终检查](./docs/GITHUB_RELEASE_CHECKLIST.md)
- [GitHub 项目文案](./docs/GITHUB_PROJECT_COPY.md)
- [安装与启动排查](./docs/TROUBLESHOOTING.md)

## 开源协议

本项目当前使用 [MIT License](./LICENSE)。

## 当前状态

当前项目适合：

- 本地演示
- 产品原型验证
- 作为可运行的功能 Demo 继续迭代

当前项目还没有完成：

- 面向公网的正式部署
- 手机安装式 PWA 配置
- 高精度 GPU 跟踪链路的稳定上线
