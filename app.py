import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

def dashboard_pertumbuhan_pdrb():
    # Memuat dan membersihkan data
    data = pd.read_csv('laju-pertumbuhan-pdrb-aceh.csv', delimiter=';')
    data.columns = data.columns.str.strip()
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Mengonversi 'tahun' dan 'laju_pertumbuhan_ekonomi' menjadi numerik
    data['tahun'] = pd.to_numeric(data['tahun'])
    data['laju_pertumbuhan_ekonomi'] = pd.to_numeric(data['laju_pertumbuhan_ekonomi'])

    # Mengganti nama kolom
    data = data.rename(columns={
        'bps_nama_provinsi': 'Provinsi',
        'laju_pertumbuhan_ekonomi': 'Laju pertumbuhan Ekonomi'
    })

    # Menghitung perbedaan persentase laju pertumbuhan dengan tahun sebelumnya
    data['Perbedaan (%)'] = data['Laju pertumbuhan Ekonomi'].diff()

    # Judul Aplikasi Streamlit dan Deskripsi
    st.title("Laju Pertumbuhan PDRB Aceh dan Prediksi 5 Tahun Ke Depan")
    st.markdown("""
    Salah satu indikator penting untuk mengetahui kondisi ekonomi di suatu negara dalam suatu periode tertentu adalah data Produk Domestik Regional Bruto (PDRB), baik atas dasar harga berlaku maupun atas dasar harga konstan. PDRB pada dasarnya merupakan jumlah nilai tambah yang dihasilkan oleh seluruh unit usaha dalam suatu negara tertentu, atau merupakan jumlah nilai barang dan jasa akhir yang dihasilkan oleh seluruh unit ekonomi. PDRB atas dasar harga berlaku menggambarkan nilai tambah barang dan jasa yang dihitung menggunakan harga yang berlaku pada setiap tahun. PDRB atas dasar harga berlaku dapat digunakan untuk melihat pergeseran dan struktur ekonomi.""")

    # Filter data berdasarkan rentang tahun yang dipilih
    year_range = (int(data['tahun'].min()), int(data['tahun'].max()))
    filtered_data = data[(data['tahun'] >= year_range[0]) & (data['tahun'] <= year_range[1])]

    # Mengonversi 'tahun' menjadi string untuk mencegah adanya tanda koma
    filtered_data['tahun'] = filtered_data['tahun'].astype(str)

    # Metrik Utama
    average_growth = filtered_data['Laju pertumbuhan Ekonomi'].mean()
    st.metric("Rata-rata Laju Pertumbuhan", f"{average_growth:.2f}%")

    # Grafik Garis untuk Pertumbuhan Ekonomi menggunakan Plotly dengan template hover kustom
    st.write("### Laju Pertumbuhan Ekonomi Seiring Waktu")
    fig = px.line(filtered_data, x='tahun', y='Laju pertumbuhan Ekonomi', markers=True,
                  title="Laju Pertumbuhan Ekonomi Seiring Waktu")

    # Mengustomisasi template hover untuk menunjukkan persentase dan perbedaan tahunan
    fig.update_traces(hovertemplate=(
        'Tahun: %{x}<br>'
        'Laju Pertumbuhan: %{y:.2f}%<br>'
        'Perbedaan dari Tahun Lalu: %{customdata:.2f}%'
    ))

    fig.update_layout(xaxis_title="Tahun", yaxis_title="Laju Pertumbuhan (%)")
    
    # Menambahkan custom data untuk hover
    fig.update_traces(customdata=filtered_data['Perbedaan (%)'])

    st.plotly_chart(fig)
    
    # Penjelasan setelah grafik pertama
    st.markdown("""
    Grafik pertama menampilkan laju pertumbuhan ekonomi tahunan dari tahun 2015 hingga 2023. Dalam periode ini, laju pertumbuhan ekonomi mengalami beberapa fluktuasi signifikan. Dimulai dari level yang lebih rendah pada tahun 2015, pertumbuhan ekonomi secara bertahap meningkat dan mencapai puncaknya sekitar 4% pada tahun 2018. Namun, terjadi penurunan tajam pada tahun 2020, yang kemungkinan besar dipengaruhi oleh pandemi COVID-19. Setelah itu, ekonomi kembali pulih dan menunjukkan stabilitas, dengan laju pertumbuhan berkisar di angka 4% pada tahun 2022 dan 2023.
    """)

    # Model Regresi Linier untuk Prediksi
    X = np.array(data['tahun']).reshape(-1, 1)
    y = data['Laju pertumbuhan Ekonomi']

    model = LinearRegression()
    model.fit(X, y)

    # Prediksi untuk 5 tahun ke depan
    future_years = np.arange(data['tahun'].max() + 1, data['tahun'].max() + 6).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    # Membuat DataFrame prediksi
    prediction_df = pd.DataFrame({
        'tahun': future_years.flatten(),
        'Laju pertumbuhan Ekonomi': future_predictions
    })

    # Menghitung perbedaan persentase tahunan pada data prediksi
    prediction_df['Perbedaan (%)'] = prediction_df['Laju pertumbuhan Ekonomi'].diff()

    # Menampilkan data prediksi
    st.write("### Prediksi Laju Pertumbuhan Ekonomi 5 Tahun Kedepan")
    fig = px.line(prediction_df, x='tahun', y='Laju pertumbuhan Ekonomi', markers=True,
                title="Prediksi Laju Pertumbuhan Ekonomi 5 Tahun Kedepan")

    fig.update_traces(hovertemplate=(
        'Tahun: %{x}<br>'
        'Laju Pertumbuhan: %{y:.2f}%<br>'
        'Perbedaan dari Tahun Lalu: %{customdata:.2f}%'
    ))

    # Menambahkan custom data untuk hover
    fig.update_traces(customdata=prediction_df['Perbedaan (%)'])

    # Mengatur x-axis
    fig.update_xaxes(tickformat='.0f', tickvals=prediction_df['tahun'].unique())

    fig.update_layout(xaxis_title="Tahun", yaxis_title="Laju Pertumbuhan (%)")

    st.plotly_chart(fig)
    
    # Penjelasan setelah grafik kedua
    st.markdown("""
    Grafik kedua memvisualisasikan prediksi laju pertumbuhan ekonomi untuk lima tahun ke depan, dari tahun 2024 hingga 2028. Berdasarkan prediksi ini, laju pertumbuhan ekonomi diperkirakan akan mengalami peningkatan yang stabil. Pada tahun 2024, laju pertumbuhan diperkirakan berada di sekitar 4,2% dan terus meningkat setiap tahunnya hingga mencapai 5,2% pada tahun 2028. Grafik ini mencerminkan ekspektasi positif terhadap perkembangan ekonomi di masa depan, dengan proyeksi pertumbuhan yang terus meningkat dalam lima tahun mendatang.
    """)


def dashboard_pdrb_per_kapita():
    # Load the data
    data = pd.read_csv('produk-domestik-regional-bruto-per-kapita-menurut-kabupaten-kota.csv', delimiter=';')

    # Convert the 'tahun' column to string to avoid automatic formatting with commas
    data['tahun'] = data['tahun'].astype(str)

    # Select relevant columns and rename them
    data = data[['bps_nama_kabupaten_kota', 'tahun', 'nilai']].rename(columns={
        'bps_nama_kabupaten_kota': 'Nama Kabupaten/Kota',
        'tahun': 'Tahun',
        'nilai': 'Nilai'
    })

    # Title
    st.title('Visualisasi Data PDRB Perkapita')

    # Sidebar for selection
    st.sidebar.header('Pilih Nama Kabupaten/Kota')
    selected_fields = st.sidebar.multiselect(
        'Nama Kabupaten/Kota:', 
        list(data['Nama Kabupaten/Kota'].unique()) + ['Semua Kabupaten/Kota']
    )
    
    # Sidebar for top selection
    st.sidebar.header('Pilih Teratas')
    top_selection = st.sidebar.radio(
        'Pilih Tipe:',
        ('None', '3 Teratas', '5 Teratas', '10 Teratas')
    )
    
    # Sidebar for bottom selection
    st.sidebar.header('Pilih Terbawah')
    bottom_selection = st.sidebar.radio(
        'Pilih Tipe:',
        ('None', '3 Terbawah', '5 Terbawah', '10 Terbawah')
    )

    # Initialize variables for data to be displayed
    combined_data = pd.DataFrame()
    avg_data = pd.DataFrame()

    # Handle data filtering for "top" and "bottom" selections
    if top_selection != 'None' or bottom_selection != 'None':
        latest_year = data['Tahun'].max()
        latest_year_data = data[data['Tahun'] == latest_year]

        if top_selection != 'None':
            if top_selection == '3 Teratas':
                selected_fields = latest_year_data.nlargest(3, 'Nilai')['Nama Kabupaten/Kota'].tolist()
            elif top_selection == '5 Teratas':
                selected_fields = latest_year_data.nlargest(5, 'Nilai')['Nama Kabupaten/Kota'].tolist()
            elif top_selection == '10 Teratas':
                selected_fields = latest_year_data.nlargest(10, 'Nilai')['Nama Kabupaten/Kota'].tolist()

        elif bottom_selection != 'None':
            if bottom_selection == '3 Terbawah':
                selected_fields = latest_year_data.nsmallest(3, 'Nilai')['Nama Kabupaten/Kota'].tolist()
            elif bottom_selection == '5 Terbawah':
                selected_fields = latest_year_data.nsmallest(5, 'Nilai')['Nama Kabupaten/Kota'].tolist()
            elif bottom_selection == '10 Terbawah':
                selected_fields = latest_year_data.nsmallest(10, 'Nilai')['Nama Kabupaten/Kota'].tolist()

    if not selected_fields:
        st.warning("Silakan pilih setidaknya satu Nama Kabupaten/Kota atau opsi top/bottom untuk menampilkan grafik.")
        return

    # Handle data filtering
    if 'Semua Kabupaten/Kota' in selected_fields:
        # Aggregate data for all fields
        filtered_data = data.copy()
        selected_fields.remove('Semua Kabupaten/Kota')  # Remove 'Semua Kabupaten/Kota' from the selected list
        
        # Compute average PDRB per capita for all fields
        avg_data = filtered_data.groupby('Tahun').agg({'Nilai': 'mean'}).reset_index()
        avg_data['Nama Kabupaten/Kota'] = 'Rata-Rata Semua Kabupaten/Kota'
        
        # Combine filtered data with average data
        combined_data = pd.concat([filtered_data, avg_data], ignore_index=True)
        
        # 1. Create chart for all Nama Kabupaten/Kota
        fig_all = px.line(
            filtered_data,
            x='Tahun',
            y='Nilai',
            color='Nama Kabupaten/Kota',
            title='PDRB Perkapita Semua Kabupaten/Kota',
            labels={'Tahun': 'Tahun', 'Nilai': 'Nilai (Juta)'},
            markers=True
        )
        st.plotly_chart(fig_all)
        
        # Explanation for the first chart
        st.write("""
        Grafik pertama menampilkan nilai PDRB perkapita untuk semua kabupaten/kota dari tahun ke tahun. Setiap garis pada grafik merepresentasikan kabupaten/kota yang berbeda. Grafik ini memungkinkan Anda untuk membandingkan kinerja dan tren dari berbagai kabupaten/kota sepanjang tahun.
        """)

        # 2. Create chart for average PDRB
        fig_avg = px.line(
            avg_data,
            x='Tahun',
            y='Nilai',
            title='Rata-Rata PDRB Perkapita Semua Kabupaten/Kota',
            labels={'Tahun': 'Tahun', 'Nilai': 'Nilai (Juta)'},
            markers=True
        )
        st.plotly_chart(fig_avg)

        # Explanation for the second chart
        st.write("""
        Grafik kedua menampilkan nilai rata-rata PDRB perkapita dari semua kabupaten/kota untuk setiap tahun. Garis tunggal pada grafik mewakili PDRB rata-rata, memberikan gambaran umum tentang kinerja ekonomi secara keseluruhan di Aceh. Grafik ini membantu memvisualisasikan tren ekonomi secara keseluruhan.
        """)

        # 3. Predict and create chart for the next 5 years
        future_years = np.arange(int(data['Tahun'].max()) + 1, int(data['Tahun'].max()) + 6)
        future_avg_data = pd.DataFrame()

        # Train a linear regression model on the average data
        model = LinearRegression()
        X = avg_data['Tahun'].astype(int).values.reshape(-1, 1)
        y = avg_data['Nilai'].values
        model.fit(X, y)

        # Predict future values
        future_predictions = model.predict(future_years.reshape(-1, 1))
        future_avg_data = pd.DataFrame({
            'Nama Kabupaten/Kota': 'Prediksi Rata-Rata',
            'Tahun': future_years.astype(str),
            'Nilai': future_predictions
        })

        # Create prediction chart
        fig_pred = px.line(
            future_avg_data, 
            x='Tahun', 
            y='Nilai', 
            title='Prediksi Rata-Rata PDRB Per Kapita 5 Tahun Kedepan',
            labels={'Tahun': 'Tahun', 'Nilai': 'Nilai (Juta)'},
            markers=True
        )
        st.plotly_chart(fig_pred)

        # Explanation for the third chart
        st.write("""
        Grafik ketiga memprediksi nilai rata-rata PDRB perkapita untuk lima tahun mendatang berdasarkan data historis. Grafik ini memberikan perkiraan tentang bagaimana PDRB rata-rata dapat berubah di masa depan.
        """)

    else:
        # Filter data by selected Nama Kabupaten/Kota
        filtered_data = data[data['Nama Kabupaten/Kota'].isin(selected_fields)]
        combined_data = filtered_data

        # Create line chart using Plotly for historical data
        fig = px.line(
            combined_data, 
            x='Tahun', 
            y='Nilai', 
            color='Nama Kabupaten/Kota',
            title='Perbandingan PDRB Antara Kabupaten/Kota', 
            labels={'Tahun': 'Tahun', 'Nilai': 'Nilai (Juta)'},
            markers=True
        )

        # Customize hover template
        fig.update_traces(hovertemplate='Nama Kabupaten/Kota: %{color}<br>Tahun: %{x}<br>Nilai: %{y:.2f} Juta')

        # Add axis labels
        fig.update_layout(xaxis_title="Tahun", yaxis_title="Nilai (Juta)")
        
        # Display the Plotly chart
        st.plotly_chart(fig)

        # Explanation for the comparison chart
        st.write("""
        Grafik ini membandingkan nilai PDRB Perkapita dari kabupaten/kota terpilih selama beberapa tahun. Grafik ini berguna untuk menganalisis kabupaten/kota tertentu dan kinerjanya relatif satu sama lain.
        """)

        # Predict the next 5 years for selected fields
        future_years = np.arange(int(data['Tahun'].max()) + 1, int(data['Tahun'].max()) + 6)
        prediction_data = pd.DataFrame()

        for field in selected_fields:
            field_data = combined_data[combined_data['Nama Kabupaten/Kota'] == field]

            if len(field_data) > 0:
                # Train a linear regression model
                model = LinearRegression()
                X = field_data['Tahun'].astype(int).values.reshape(-1, 1)
                y = field_data['Nilai'].values
                model.fit(X, y)

                # Predict future values
                future_predictions = model.predict(future_years.reshape(-1, 1))
                field_prediction_data = pd.DataFrame({
                    'Nama Kabupaten/Kota': field,
                    'Tahun': future_years.astype(str),
                    'Nilai': future_predictions
                })

                prediction_data = pd.concat([prediction_data, field_prediction_data], ignore_index=True)

        # Create prediction chart
        fig_pred = px.line(
            prediction_data, 
            x='Tahun', 
            y='Nilai', 
            color='Nama Kabupaten/Kota',
            title='Prediksi PDRB Per Kapita 5 Tahun Kedepan',
            labels={'Tahun': 'Tahun', 'Nilai': 'Nilai (Juta)'},
            markers=True
        )
        st.plotly_chart(fig_pred)

        # Explanation for the prediction chart
        st.write("""
        Grafik ini memprediksi nilai PDRB Perkapita dari kabupaten/kota yang dipilih selama lima tahun mendatang. Prediksi ini didasarkan pada tren historis dan membantu memperkirakan bagaimana PDRB kabupaten/kota tertentu mungkin berkembang di masa depan.
        """)

            
def dashboard_pdrb_adhb_aceh():
    # Load the data
    data = pd.read_csv('pdrb-adhb-aceh-tahun-2010-2023.csv', delimiter=';')

    # Convert the 'tahun' column to string to avoid automatic formatting with commas
    data['tahun'] = data['tahun'].astype(str)

    # Select relevant columns and rename them
    data = data[['lapangan_usaha', 'tahun', 'pdrb']].rename(columns={
        'lapangan_usaha': 'Lapangan Usaha',
        'tahun': 'Tahun',
        'pdrb': 'Nilai PDRB'
    })

    # Title
    st.title('Visualisasi Data PDRB ADHB')

    # Sidebar for selection
    st.sidebar.header('Pilih Lapangan Usaha')
    selected_fields = st.sidebar.multiselect(
        'Lapangan Usaha:', 
        list(data['Lapangan Usaha'].unique()) + ['Semua Lapangan Usaha']
    )
    
    # Sidebar for top selection
    st.sidebar.header('Pilih Teratas')
    top_selection = st.sidebar.radio(
        'Pilih Tipe:',
        ('None', '3 Teratas', '5 Teratas', '10 Teratas')
    )
    
    # Sidebar for bottom selection
    st.sidebar.header('Pilih Terbawah')
    bottom_selection = st.sidebar.radio(
        'Pilih Tipe:',
        ('None', '3 Terbawah', '5 Terbawah', '10 Terbawah')
    )

    # Initialize variables for data to be displayed
    combined_data = pd.DataFrame()
    avg_data = pd.DataFrame()

    # Handle data filtering for "top" and "bottom" selections
    if top_selection != 'None' or bottom_selection != 'None':
        latest_year = data['Tahun'].max()
        latest_year_data = data[data['Tahun'] == latest_year]

        if top_selection != 'None':
            if top_selection == '3 Teratas':
                selected_fields = latest_year_data.nlargest(3, 'Nilai PDRB')['Lapangan Usaha'].tolist()
            elif top_selection == '5 Teratas':
                selected_fields = latest_year_data.nlargest(5, 'Nilai PDRB')['Lapangan Usaha'].tolist()
            elif top_selection == '10 Teratas':
                selected_fields = latest_year_data.nlargest(10, 'Nilai PDRB')['Lapangan Usaha'].tolist()

        elif bottom_selection != 'None':
            if bottom_selection == '3 Terbawah':
                selected_fields = latest_year_data.nsmallest(3, 'Nilai PDRB')['Lapangan Usaha'].tolist()
            elif bottom_selection == '5 Terbawah':
                selected_fields = latest_year_data.nsmallest(5, 'Nilai PDRB')['Lapangan Usaha'].tolist()
            elif bottom_selection == '10 Terbawah':
                selected_fields = latest_year_data.nsmallest(10, 'Nilai PDRB')['Lapangan Usaha'].tolist()

    if not selected_fields:
        st.warning("Silakan pilih setidaknya satu Lapangan Usaha atau opsi top/bottom untuk menampilkan grafik.")
        return

    # Handle data filtering
    if 'Semua Lapangan Usaha' in selected_fields:
        # Aggregate data for all fields
        filtered_data = data.copy()
        selected_fields.remove('Semua Lapangan Usaha')  # Remove 'Semua Lapangan Usaha' from the selected list
        
        # Compute average PDRB per capita for all fields
        avg_data = filtered_data.groupby('Tahun').agg({'Nilai PDRB': 'mean'}).reset_index()
        avg_data['Lapangan Usaha'] = 'Rata-Rata Semua Lapangan Usaha'
        
        # Combine filtered data with average data
        combined_data = pd.concat([filtered_data, avg_data], ignore_index=True)
        
        # 1. Create chart for all Lapangan Usaha
        fig_all = px.line(
            filtered_data,
            x='Tahun',
            y='Nilai PDRB',
            color='Lapangan Usaha',
            title='PDRB Semua Lapangan Usaha',
            labels={'Tahun': 'Tahun', 'Nilai PDRB': 'Nilai (Miliar)'},
            markers=True
        )
        st.plotly_chart(fig_all)
        
        # Explanation for the first chart
        st.write("""
        Grafik pertama menampilkan nilai PDRB Perkapita untuk semua sektor (Lapangan Usaha) dari tahun ke tahun. Setiap garis pada grafik merepresentasikan sektor ekonomi yang berbeda. Grafik ini memungkinkan Anda untuk membandingkan kinerja dan tren dari berbagai sektor ekonomi sepanjang tahun.
        """)

        # 2. Create chart for average PDRB
        fig_avg = px.line(
            avg_data,
            x='Tahun',
            y='Nilai PDRB',
            title='Rata-Rata PDRB Semua Lapangan Usaha',
            labels={'Tahun': 'Tahun', 'Nilai PDRB': 'Nilai (Miliar)'},
            markers=True
        )
        st.plotly_chart(fig_avg)

        # Explanation for the second chart
        st.write("""
        Grafik kedua menampilkan nilai rata-rata PDRB Perkapita dari semua sektor untuk setiap tahun. Garis tunggal pada grafik mewakili PDRB rata-rata, memberikan gambaran umum tentang kinerja ekonomi secara keseluruhan di Aceh. Grafik ini membantu memvisualisasikan tren ekonomi secara keseluruhan.
        """)

        # 3. Predict and create chart for the next 5 years
        future_years = np.arange(int(data['Tahun'].max()) + 1, int(data['Tahun'].max()) + 6)
        future_avg_data = pd.DataFrame()

        # Train a linear regression model on the average data
        model = LinearRegression()
        X = avg_data['Tahun'].astype(int).values.reshape(-1, 1)
        y = avg_data['Nilai PDRB'].values
        model.fit(X, y)

        # Predict future values
        future_predictions = model.predict(future_years.reshape(-1, 1))
        future_avg_data = pd.DataFrame({
            'Lapangan Usaha': 'Prediksi Rata-Rata',
            'Tahun': future_years.astype(str),
            'Nilai': future_predictions
        })

        # Create prediction chart
        fig_pred = px.line(
            future_avg_data, 
            x='Tahun', 
            y='Nilai', 
            title='Prediksi Rata-Rata PDRB Per Kapita 5 Tahun Kedepan',
            labels={'Tahun': 'Tahun', 'Nilai PDRB Per Kapita': 'Nilai (Miliar)'},
            markers=True
        )
        st.plotly_chart(fig_pred)

        # Explanation for the third chart
        st.write("""
        Grafik ketiga memprediksi nilai rata-rata PDRB untuk lima tahun mendatang berdasarkan data historis. Grafik ini memberikan perkiraan tentang bagaimana PDRB rata-rata dapat berubah di masa depan.
        """)

    else:
        # Filter data by selected Lapangan Usaha
        filtered_data = data[data['Lapangan Usaha'].isin(selected_fields)]
        combined_data = filtered_data

        # Create line chart using Plotly for historical data
        fig = px.line(
            combined_data, 
            x='Tahun', 
            y='Nilai PDRB', 
            color='Lapangan Usaha',
            title='Perbandingan PDRB Antara Lapangan Usaha', 
            labels={'Tahun': 'Tahun', 'Nilai PDRB Per Kapita': 'Nilai (Miliar)'},
            markers=True
        )

        # Customize hover template
        fig.update_traces(hovertemplate='Lapangan Usaha: %{color}<br>Tahun: %{x}<br>Nilai PDRB: %{y:.2f} Miliar')

        # Add axis labels
        fig.update_layout(xaxis_title="Tahun", yaxis_title="Nilai PDRB(Miliar)")
        
        # Display the Plotly chart
        st.plotly_chart(fig)

        # Explanation for the comparison chart
        st.write("""
        Grafik ini membandingkan nilai PDRB dari sektor-sektor terpilih selama beberapa tahun. Grafik ini berguna untuk menganalisis sektor-sektor tertentu dan kinerjanya relatif satu sama lain.
        """)

        # Predict the next 5 years for selected fields
        future_years = np.arange(int(data['Tahun'].max()) + 1, int(data['Tahun'].max()) + 6)
        prediction_data = pd.DataFrame()

        for field in selected_fields:
            field_data = combined_data[combined_data['Lapangan Usaha'] == field]

            if len(field_data) > 0:
                # Train a linear regression model
                model = LinearRegression()
                X = field_data['Tahun'].astype(int).values.reshape(-1, 1)
                y = field_data['Nilai PDRB'].values
                model.fit(X, y)

                # Predict future values
                future_predictions = model.predict(future_years.reshape(-1, 1))
                field_prediction_data = pd.DataFrame({
                    'Lapangan Usaha': field,
                    'Tahun': future_years.astype(str),
                    'Nilai PDRB': future_predictions
                })

                prediction_data = pd.concat([prediction_data, field_prediction_data], ignore_index=True)

        # Create prediction chart
        fig_pred = px.line(
            prediction_data, 
            x='Tahun', 
            y='Nilai PDRB', 
            color='Lapangan Usaha',
            title='Prediksi PDRB Per Kapita 5 Tahun Kedepan',
            labels={'Tahun': 'Tahun', 'Nilai PDRB Per Kapita': 'Nilai (Miliar)'},
            markers=True
        )
        st.plotly_chart(fig_pred)

        # Explanation for the prediction chart
        st.write("""
        Grafik ini memprediksi nilai PDRB untuk lima tahun mendatang bagi sektor-sektor terpilih. Grafik ini berguna untuk mengidentifikasi kemungkinan tren pertumbuhan atau penurunan dalam sektor-sektor tertentu.
        """)
         
# Sidebar untuk memilih dashboard
dashboard = st.sidebar.selectbox(
    "Pilih Dashboard",
    ("Laju Pertumbuhan PDRB", "PDRB Per Kapita Aceh", "PDRB ADHB ACEH")
)

# Menampilkan dashboard yang dipilih
if dashboard == "Laju Pertumbuhan PDRB":
    dashboard_pertumbuhan_pdrb()
elif dashboard == "PDRB Per Kapita Aceh":
    dashboard_pdrb_per_kapita()
elif dashboard == "PDRB ADHB ACEH":
    dashboard_pdrb_adhb_aceh()
    
