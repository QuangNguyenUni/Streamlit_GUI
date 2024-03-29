import streamlit as st 

import pandas as pd    
import matplotlib.pyplot as plt     
import seaborn as sns
st.title("Trung Tâm Tin Học")
menu = ["Giới thiệu project", "Tổng quan project Sentiment Analysis", "Demo"]
choice = st.sidebar.selectbox('Danh mục', menu)

if choice == 'Giới thiệu project':    
    st.write("""## Giới thiệu về Project Sentiment Analysis""")
    st.write("""
    Phân tích tâm trạng là quá trình phát hiện tâm trạng tích cực hoặc tiêu cực trong văn bản. Thường được sử dụng bởi doanh nghiệp để phát hiện tâm trạng trong dữ liệu mạng xã hội, đánh giá uy tín thương hiệu và hiểu biết về khách hàng.
    """)
    
    st.subheader("Mục tiêu project: ")
    st.write("""
    - Phân loại một đánh giá là positive hay là negative.
    - Đánh giá tổng quan một nhà hàng.""")
    st.subheader("Các bước thực hiện: ")
    st.write("""
    1. Xem và tiền xử lý dữ liệu
    2. Build model với sklearn và pyspark
    3. Đánh giá kết quả và chọn model phù hợp
    4. Lưu model đã chọn
    5. Xây dựng file đánh giá overview cho nhà hàng
    6. Xây dựng giao diện
    7. Báo cáo kết quả""")
elif choice == 'Tổng quan project Sentiment Analysis':
    st.subheader("1. Đọc và tiền xử lý dữ liệu")
    st.write("      2 Dataframe ban đầu: ")
    df_1 = pd.read_csv("1_Restaurants.csv")
    df_2 = pd.read_csv("2_Reviews.csv")
    st.write(df_1.head())
    st.write(df_2.head())
    st.write("      Dataframe sau khi sử lý và cho model học")
    df = pd.read_csv("Sentiment_data_after_processing.csv")
    df = df[['Comment_new', 'label']]
    head_df = df.head()
    st.write(head_df)
    st.subheader("2. Train model and report")
    st.write("Model tốt nhất là random forest của sklearn. Và các thông số như sau: ")
    st.write("##### Confusion matrix: ")
    st.image('confusion_matrix.png', caption='Confusion Matrix')
    st.write("##### Classification report: ")
    # Display the classification report text
    with open('classification_report.txt', 'r') as file:
        class_report_text = file.read()
    st.text('Classification Report:')
    st.text(class_report_text)
    st.write("##### ROC Curve")
    st.image("Random_Forest_ROC_Curve.png")
    st.subheader("Wordcloud cho Positive: ")
    st.image("positive_wordcloud.png")
    
    st.subheader("Wordcloud cho Negative: ")
    st.image("negative_wordcloud.png")
elif choice == 'Demo':
    st.subheader("Gợi ý điều khiển project 1")
    # Cho người dùng chọn nhập dữ liệu hoặc upload file
    type = st.radio("Chọn cách nhập dữ liệu", options=["Nhập dữ liệu", "Upload file", "Đánh giá nhà hàng"])
    # Nếu người dùng chọn nhập dữ liệu vào text area
    # if type == "Nhập dữ liệu":
    #     st.subheader("Nhập dữ liệu")
    #     content = st.text_area("Nhập ý kiến:")
    #     # Nếu người dùng nhập dữ liệu đưa content này vào thành 1 dòng trong DataFrame
    #     if content:
    #         df = pd.DataFrame([content], columns=["Ý kiến"])
    #         st.dataframe(df)
    #         st.table(df)
    import pickle

# Load the pipeline model
    # Load the pipeline model
    with open('forest_pipeline_model.pkl', 'rb') as file:
        forest_pipeline_model = pickle.load(file)

    # Check if the user selects "Nhập dữ liệu" option
    if type == "Nhập dữ liệu":
        st.subheader("Nhập dữ liệu")
        
        # Allow the user to input text
        user_input = st.text_input("Nhập nhận xét: ")
        
        # Convert user input into the desired format
        test = [user_input]
        
        # Make predictions using the loaded model
        submitted_button = st.button("Dự đoán cảm xúc: ")
        if forest_pipeline_model and user_input and submitted_button:
            test_output = forest_pipeline_model.predict(test)
            
            # Display the predicted sentiment
            if test_output == [0]:
                st.write("Negative")
            elif test_output == [1]:
                st.write("Positive")
    
    # elif type == "Upload file":
    #     st.subheader("Upload file")
    #     # Upload file
    #     uploaded_file = st.file_uploader("Chọn file dữ liệu", type=["csv", "txt", "xlsx"])
    #     if uploaded_file is not None:
    #         # Đọc file dữ liệu
    #         df = pd.read_excel(uploaded_file)
    #         st.write(df)
    elif type == "Upload file":
        st.subheader("Upload file")
        # Upload file
        uploaded_file = st.file_uploader("Chọn file dữ liệu", type=["csv", "txt", "xlsx"])
        # if uploaded_file is not None:            
        #     df = pd.read_excel(uploaded_file)
        #     st.write(df)
            
        #     # Make predictions using the loaded model
        #     if forest_pipeline_model:
        #         predictions = forest_pipeline_model.predict(df['Comment'])  # Assuming 'Ý kiến' is the column containing text data
        #         df['Predicted Sentiments'] = predictions

        #         # Display the predictions
        #         st.write("Predicted Sentiments:")
        #         st.write(df)
        # Make predictions for each comment
        if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
            if uploaded_file.type == 'text/csv':
                df_comments = pd.read_csv(uploaded_file)
            elif uploaded_file.type == 'text/plain':
                df_comments = pd.read_csv(uploaded_file, delimiter='\t')
            elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                df_comments = pd.read_excel(uploaded_file)
            else:
                st.write("Unsupported file format")

        # Make predictions for each comment
        predictions = []
        for comment in df_comments['Comment']:
            # Convert the comment into the desired format
            test = [comment]
            
            # Make predictions using the loaded model
            if forest_pipeline_model:
                prediction = forest_pipeline_model.predict(test)
                
                # Map predicted values to "Positive" or "Negative"
                sentiment = "Positive" if prediction[0] == 1 else "Negative"
                predictions.append(sentiment)  # Append the predicted sentiment to the list

        # Add the predictions to the DataFrame
        df_comments['Predicted Sentiment'] = predictions

        # Display the DataFrame with predictions
        st.write(df_comments)

    elif type == "Đánh giá nhà hàng":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        restaurants = pd.read_csv("restaurants_and_reviews")
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        def calculate_average_rating_and_wordcloud(df, restaurant_id):
            # Filter dataframe based on provided restaurant ID
            restaurant_df = df[df['IDRestaurant'] == int(restaurant_id)]
            
            # Check if the restaurant ID exists in the dataframe
            if restaurant_df.empty:
                st.write(f"No data found for restaurant with ID {restaurant_id}")
                return
            
            # Calculate average rating
            average_rating = restaurant_df['Rating'].mean()
            st.write(f"Average Rating for Restaurant ID {restaurant_id}: {average_rating:.2f}")
            
            # Generate word cloud for comments
            comments = ' '.join(restaurant_df['Comment'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments)
            
            # Plot the word cloud
            st.write("Word Cloud for Comments:")
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Word Cloud for Comments (Restaurant ID: {restaurant_id})")
            st.pyplot()
        if type == "Đánh giá nhà hàng":
            st.subheader("Đánh giá nhà hàng")
    
            # Allow the user to input restaurant ID
            user_input_id = st.text_input("Nhập ID nhà hàng: ")
    
            # Calculate average rating and display word cloud
        submitted_button_2 = st.button("Đánh giá")
        if user_input_id and submitted_button_2:
            calculate_average_rating_and_wordcloud(restaurants, user_input_id)
