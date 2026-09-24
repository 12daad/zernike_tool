# Zernike Tool

## 1. Introduction

Zernike Tool is a NumPy-based Python toolkit for rectangular Zernike polynomial evaluation, coefficient calculation, and wavefront reconstruction.  
It also provides a Windows GUI for batch image calibration, measured gray–phase response mapping, preview, and export.

## 2. Quick Start

Requires **Python >= 3.14**.

### 2.1 Install requirements

On Windows, install the project using PowerShell/Command Prompt or [`uv`](https://docs.astral.sh/uv/getting-started/installation).

#### Using PowerShell/Command Prompt

```text
git clone https://github.com/12daad/zernike_tool.git
cd zernike_tool
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install .
```

#### Using uv

```text
git clone https://github.com/12daad/zernike_tool.git
cd zernike_tool
uv sync
```

### 2.2 Start Up

#### Using PowerShell/Command Prompt

```text
.\.venv\Scripts\activate
python -m zernike_tool.app
```

#### Using uv

```text
uv run python -m zernike_tool.app
```
