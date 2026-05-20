import speedtest

st = speedtest.Speedtest()

print("Finding best server...")
st.get_best_server()

print("Testing download...")
download_speed = st.download() / 1_000_000

print("Testing upload...")
upload_speed = st.upload() / 1_000_000

ping = st.results.ping

print(f"Download Speed: {download_speed:.2f} Mbps")
print(f"Upload Speed: {upload_speed:.2f} Mbps")
print(f"Ping: {ping:.2f} ms")