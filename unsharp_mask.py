import cv2 #xulyanh
import numpy as np #xulymatran
import streamlit as st #giaodien

uploaded_image = st.file_uploader("Upload ảnh",type = ["jpg","png","jpeg"]) #nut de nguoi dung up anh tu may tinh

k = st.sidebar.slider("Độ nét ảnh", min_value = 0.0, max_value = 5.0, value = 1.0, step = 0.5)
if uploaded_image is not None:
    #doc du lieu tho duoc up len -> phan loai thanh byte -> chuyen byte thanh mang
    file_bytes = np.asanyarray(bytearray(uploaded_image.read()), dtype = np.uint8)
    # dua mang ve dang ma tran mau
    image = cv2.imdecode(file_bytes, 1)

    if image is None:
        st.error("Không có ảnh")
    else:

        anh_goc = image
        col1, col2 = st.columns(2) 
        
        with col1:
            st.subheader("Ảnh gốc")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner("Hold on bro"):

            gaussian = np.array([[0.0625, 0.125, 0.0625],
                               [0.125, 0.25, 0.125],
                               [0.0625, 0.125, 0.0625]])
        
            after_blur = cv2.filter2D(image, -1, gaussian)

            output = anh_goc.astype(np.float32) + k*(anh_goc.astype(np.float32) - after_blur.astype(np.float32))
            output = np.clip(output, 0, 255).astype(np.uint8)

            with col2:
                st.subheader("done roi ni'")
                # chuyen output tu bgr sang rgb
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                st.image(output_rgb, use_container_width=True)

                
            


        