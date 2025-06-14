import pandas as pd

def csv_to_xlsx(csv_file_path, xlsx_file_path):
    """
    Mengonversi file CSV ke file XLSX.

    Args:
        csv_file_path (str): Path lengkap ke file CSV input.
        xlsx_file_path (str): Path lengkap untuk menyimpan file XLSX output.
    """
    try:
        # Membaca file CSV
        df = pd.read_csv(csv_file_path)

        # Menulis DataFrame ke file XLSX
        df.to_excel(xlsx_file_path, index=False)
        print(f"File '{csv_file_path}' berhasil dikonversi menjadi '{xlsx_file_path}'")
    except FileNotFoundError:
        print(f"Error: File CSV tidak ditemukan di '{csv_file_path}'")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# Contoh penggunaan:
if __name__ == "__main__":
    # Ganti dengan path file CSV Anda
    input_csv = "dataset/dataset_cyberbullying.csv"
    # Ganti dengan nama file XLSX yang ingin Anda buat
    output_xlsx = "dataset/dataset_cyberbullying.xlsx"

    # Membuat contoh file CSV (opsional, jika Anda belum punya file CSV)
    # sample_data = {
    #     'Nama': ['Andi', 'Budi', 'Citra'],
    #     'Usia': [30, 24, 28],
    #     'Kota': ['Jakarta', 'Bandung', 'Surabaya']
    # }
    # sample_df = pd.DataFrame(sample_data)
    # sample_df.to_csv(input_csv, index=False)
    # print(f"Contoh file CSV '{input_csv}' telah dibuat.")

    csv_to_xlsx(input_csv, output_xlsx)