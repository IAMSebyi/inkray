from fastapi import HTTPException, UploadFile


async def read_image_upload(file: UploadFile, max_bytes: int) -> bytes:
    raw_bytes = await file.read(max_bytes + 1)
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds the file size limit.")
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
    return raw_bytes
