import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sounddevice as sd
import numpy as np
import os
import librosa
import soundfile as sf
from sklearn.decomposition import PCA
import threading
import time

# ==================== VARIÁVEIS GLOBAIS ====================
is_playing = False
is_paused = False
stop_flag = False
current_position = 0
audio_data = None
audio_sr = 44100
stream = None
playback_thread = None
total_duration = 0
volume_level = 0.06  # Volume inicial (6%)
compression_quality = "media-alta" # Variável global para armazenar a qualidade de compressão

# ==================== FUNÇÕES DE COMPRESSÃO E DESCOMPRESSÃO ====================

def pega_subtipo_do_audio (caminho_do_audio):
    # Retorna o subtipo do audio original (bits por amostra)
    inf = sf.info(caminho_do_audio)
    return inf.subtype

def parametros_dinamicos(y, sample_rate, quality_level):
    sample_rate_ref = 96000

    if quality_level == "alta":
        N_FFT_ref = 100
        HOP_LENGTH_ref = 80
        N_COMPONENTS_ref = 25
    if quality_level == "media-alta":
        N_FFT_ref = 300
        HOP_LENGTH_ref = 235
        N_COMPONENTS_ref = 62
    # Já que a qualidade media-baixa usa 8 bits, vamos aumentar os parâmetros para compensar a diminuição de bits
    elif quality_level == "media-baixa":
        N_FFT_ref = 300
        HOP_LENGTH_ref = 120
        N_COMPONENTS_ref = 90
    elif quality_level == "baixa":
        N_FFT_ref = 300
        HOP_LENGTH_ref = 235
        N_COMPONENTS_ref = 62
    else:
        # Padrão, caso algo dê errado
        N_FFT_ref = 100
        HOP_LENGTH_ref = 80
        N_COMPONENTS_ref = 25

    fator = sample_rate / sample_rate_ref

    N_FFT = int(np.round(N_FFT_ref * fator))
    HOP_LENGTH = int(np.round(HOP_LENGTH_ref * fator))
    N_COMPONENTS = int(np.round(N_COMPONENTS_ref * fator))

    if N_FFT % 2 != 0:
        N_FFT += 1

    return N_FFT, HOP_LENGTH, N_COMPONENTS

def compressao_audio_PCA(caminho_entrada_do_audio, caminho_saida_do_audio, quality_level, progress_callback=None):
    # Carrega o áudio
    base, ext = os.path.splitext(caminho_entrada_do_audio)

    if progress_callback: progress_callback(10)

    # Se o áudio não for WAV, converte para WAV temporariamente
    if ext.lower() != '.wav':
        wav_copy_path = f"{base}_converted.wav"
        y_temp, sample_rate = librosa.load(caminho_entrada_do_audio, sr=None, mono=False)
        sf.write(wav_copy_path, y_temp.T, sample_rate, subtype='PCM_16')
        caminho_entrada_do_audio = wav_copy_path
    else:
        y_temp, sample_rate = librosa.load(caminho_entrada_do_audio, sr=None, mono=False)


    if progress_callback: progress_callback(20)

    # Agora, carregamos o áudio (convertido ou já WAV)
    y_stereo, sample_rate = librosa.load(caminho_entrada_do_audio, sr=None, mono=False)
    if y_stereo.ndim == 1:
        y_stereo = np.vstack((y_stereo, y_stereo))  # Duplicamos o canal se for mono

    # Descobre subtipo do áudio original
    subtipo = pega_subtipo_do_audio(caminho_entrada_do_audio)

    # Se a qualidade for baixa, muda o subtipo para 8 bits
    if quality_level == "baixa" or quality_level == "media-baixa":
        subtipo = 'PCM_U8'

    if progress_callback: progress_callback(30)

    # Calcula parâmetros dinâmicos
    n_fft, tam_pulo, n_componentes = parametros_dinamicos(y_stereo, sample_rate, quality_level)

    # Criamos variáveis para armazenar os canais direito e esquerdo
    y_left, y_right = y_stereo[0], y_stereo[1]

    if progress_callback: progress_callback(40)

    # Usamos Transformada de Fourier para obter espectrogramas
    Espectrograma_esquerdo = librosa.stft(y_left, n_fft=n_fft, hop_length=tam_pulo)
    Espectrograma_direito = librosa.stft(y_right, n_fft=n_fft, hop_length=tam_pulo)

    if progress_callback: progress_callback(50)

    # Concatenamos os espectrogramas em uma matriz 2D
    X_left = np.concatenate((np.real(Espectrograma_esquerdo.T), np.imag(Espectrograma_esquerdo.T)), axis=1)
    X_right = np.concatenate((np.real(Espectrograma_direito.T), np.imag(Espectrograma_direito.T)), axis=1)

    if progress_callback: progress_callback(60)

    # Aplicamos o PCA separadamente no canal esquerdo
    pca_left = PCA(n_components=n_componentes)
    pca_left.fit(X_left)
    X_left_comprimido = pca_left.transform(X_left)

    if progress_callback: progress_callback(70)

    # Aplicamos o PCA separadamente no canal direito
    pca_right = PCA(n_components=n_componentes)
    pca_right.fit(X_right)
    X_right_comprimido = pca_right.transform(X_right)

    if progress_callback: progress_callback(80)

    # Agora que temos os espectrogramas comprimidos podemos salvar os dados comprimidos
    # Normaliza os dados comprimidos para o intervalo [-1, 1]
    # Se for qualidade baixa ou média-baixa, a compressão será para int8
    if quality_level == "baixa" or quality_level == "media-baixa":
        print(f"Compressão em {quality_level} 8 bits")
        max_abs_left = np.max(np.abs(X_left_comprimido))
        max_abs_right = np.max(np.abs(X_right_comprimido))

        escala_left = 127 / max_abs_left if max_abs_left > 0 else 1.0
        escala_right = 127 / max_abs_right if max_abs_right > 0 else 1.0

        X_left_comprimido_int = (X_left_comprimido * escala_left).astype(np.int8)
        X_right_comprimido_int = (X_right_comprimido * escala_right).astype(np.int8)
    else:
        print(f"Compressão em {quality_level} 16 bits")
        max_abs_left = np.max(np.abs(X_left_comprimido))
        max_abs_right = np.max(np.abs(X_right_comprimido))

        escala_left = 32767 / max_abs_left if max_abs_left > 0 else 1.0
        escala_right = 32767 / max_abs_right if max_abs_right > 0 else 1.0

        X_left_comprimido_int = (X_left_comprimido * escala_left).astype(np.int16)
        X_right_comprimido_int = (X_right_comprimido * escala_right).astype(np.int16)

    np.savez_compressed(caminho_saida_do_audio,
                        X_left_comprimido=X_left_comprimido_int,
                        X_right_comprimido=X_right_comprimido_int,
                        escala_left=escala_left,
                        escala_right=escala_right,
                        componentes_left=pca_left.components_,
                        componentes_right=pca_right.components_,
                        media_pca_left=pca_left.mean_,
                        media_pca_right=pca_right.mean_,
                        sample_rate=sample_rate,
                        tam_pulo=tam_pulo,
                        formato_original_left=Espectrograma_esquerdo.T.shape,
                        formato_original_right=Espectrograma_direito.T.shape,
                        subtipo=subtipo,
                        quality_level_saved=quality_level) # Salva o nível de qualidade usado
    
    if progress_callback: progress_callback(100)

def descompressao_audio_PCA(caminho_entrada_do_audio, caminho_saida_do_audio, progress_callback=None):
    # Carrega os dados comprimidos
    dados = np.load(caminho_entrada_do_audio)

    if progress_callback: progress_callback(10)

    quality_level_saved = str(dados['quality_level_saved'])

    # Reconverte os dados comprimidos de int16/int8 para float32
    escala_left = float(dados['escala_left'])
    escala_right = float(dados['escala_right'])

    X_left_comprimido = dados['X_left_comprimido'].astype(np.float32) / escala_left
    X_right_comprimido = dados['X_right_comprimido'].astype(np.float32) / escala_right

    componentes_left = dados['componentes_left']
    componentes_right = dados['componentes_right']
    media_pca_left = dados['media_pca_left']
    media_pca_right = dados['media_pca_right']

    sample_rate = int(dados['sample_rate'])
    tam_pulo = int(dados['tam_pulo'])
    subtipo = str(dados['subtipo'])

    formato_original_left = tuple(dados['formato_original_left'])
    formato_original_right = tuple(dados['formato_original_right'])

    if progress_callback: progress_callback(30)

    # Recria o PCA para os canais
    pca_left = PCA(n_components=componentes_left.shape[0])
    pca_left.components_ = componentes_left
    pca_left.mean_ = media_pca_left

    pca_right = PCA(n_components=componentes_right.shape[0])
    pca_right.components_ = componentes_right
    pca_right.mean_ = media_pca_right

    if progress_callback: progress_callback(50)

    # Reconstrói os espectrogramas a partir da transformação inversa do PCA
    X_left_reconstruido = pca_left.inverse_transform(X_left_comprimido)
    X_right_reconstruido = pca_right.inverse_transform(X_right_comprimido)

    if progress_callback: progress_callback(70)

    # Separa parte real e imaginária
    metade = X_left_reconstruido.shape[1] // 2
    Espectrograma_esquerdo_rec = X_left_reconstruido[:, :metade] + 1j * X_left_reconstruido[:, metade:]
    Espectrograma_direito_rec = X_right_reconstruido[:, :metade] + 1j * X_right_reconstruido[:, metade:]

    Espectrograma_esquerdo_rec = Espectrograma_esquerdo_rec.reshape(formato_original_left).T
    Espectrograma_direito_rec = Espectrograma_direito_rec.reshape(formato_original_right).T

    if progress_callback: progress_callback(80)

    # Reconstrói os sinais de tempo com ISTFT
    y_left_rec = librosa.istft(Espectrograma_esquerdo_rec, hop_length=tam_pulo)
    y_right_rec = librosa.istft(Espectrograma_direito_rec, hop_length=tam_pulo)

    # Ajusta para mesmo comprimento
    min_len = min(len(y_left_rec), len(y_right_rec))
    y_stereo_rec = np.vstack((y_left_rec[:min_len], y_right_rec[:min_len]))

    # Normaliza para faixa [-1.0, 1.0], caso esteja fora
    y_stereo_rec = np.clip(y_stereo_rec, -1.0, 1.0)

    if subtipo.upper() == 'PCM_U8':
        # Converte para int16 antes de salvar como PCM_U8, pois soundfile não aceita int8 diretamente para esse subtipo.
        # Os valores serão escalados para ocupar a faixa de 16 bits, mas mantendo a resolução de 8 bits.
        y_stereo_rec_int16 = (y_stereo_rec * 32767).astype(np.int16)
        sf.write(caminho_saida_do_audio, y_stereo_rec_int16.T, sample_rate, subtype='PCM_16') # Salva como PCM_16
    elif subtipo.upper() in ['PCM_24', 'PCM_32', 'FLOAT', 'DOUBLE']:
        # Escala para int16 para salvar em 16-bit
        y_stereo_rec_int16 = (y_stereo_rec * 32767).astype(np.int16)
        sf.write(caminho_saida_do_audio, y_stereo_rec_int16.T, sample_rate, subtype='PCM_16')
    else:
        # Mantém o subtipo original se já for 16-bit ou menor (exceto PCM_U8 já tratado)
        sf.write(caminho_saida_do_audio, y_stereo_rec.T, sample_rate, subtype=subtipo)
    
    if progress_callback: progress_callback(100)

# ==================== FUNÇÕES DA INTERFACE ====================

def update_progress_bar(bar, value):
    bar['value'] = value
    root.update_idletasks()

def selecionar_audio_compressao():
    global input_audio_path
    input_audio_path = filedialog.askopenfilename(
        title="Selecione o áudio para compressão",
        filetypes=[("Arquivos de áudio", "*.wav *.mp3 *.flac")]
    )
    if input_audio_path:
        lbl_compressao_input.config(text=os.path.basename(input_audio_path))

def set_compression_quality():
    global compression_quality
    compression_quality = quality_var.get()

def executar_compressao():
    if not input_audio_path:
        messagebox.showerror("Erro", "Selecione um arquivo primeiro!")
        return
    output_path = filedialog.asksaveasfilename(
        title="Salvar áudio comprimido",
        defaultextension=".npz",
        filetypes=[("NumPy Zip", "*.npz")]
    )
    if output_path:
        global progress_bar_compress
        progress_bar_compress.pack(padx=5, pady=5, fill='x')
        progress_bar_compress['value'] = 0
        print(f"Executando compressão com qualidade: {compression_quality}")
        threading.Thread(target=lambda: _executar_compressao_thread(output_path, compression_quality)).start()

def _executar_compressao_thread(output_path, quality_level):
     try:
        compressao_audio_PCA(input_audio_path, output_path, quality_level, lambda val: update_progress_bar(progress_bar_compress, val))
     except Exception as e:
        messagebox.showerror("Erro na compressão", str(e))
     finally:
        progress_bar_compress.pack_forget()

def selecionar_comprimido():
    global compressed_path
    compressed_path = filedialog.askopenfilename(
        title="Selecione o áudio comprimido",
        filetypes=[("NumPy Zip", "*.npz")]
    )
    if compressed_path:
        lbl_decompress_input.config(text=os.path.basename(compressed_path))

def executar_descompressao():
    if not compressed_path:
        messagebox.showerror("Erro", "Selecione um arquivo comprimido!")
        return
    output_path = filedialog.asksaveasfilename(
        title="Salvar áudio reconstruído",
        defaultextension=".wav",
        filetypes=[("WAV", "*.wav")]
    )
    if output_path:
        progress_bar_decompress.pack(padx=5, pady=5, fill='x')
        progress_bar_decompress['value'] = 0
        threading.Thread(target=lambda: _executar_descompressao_thread(output_path)).start()

def _executar_descompressao_thread(output_path):
    try:
        descompressao_audio_PCA(compressed_path, output_path, lambda val: update_progress_bar(progress_bar_decompress, val))
        messagebox.showinfo("Sucesso", f"Áudio descomprimido salvo em:\n{output_path}")
    except Exception as e:
        messagebox.showerror("Erro na descompressão", str(e))
    finally:
        progress_bar_decompress.pack_forget()

def selecionar_para_play():
    global play_compressed_path, audio_data, audio_sr, total_duration
    play_compressed_path = filedialog.askopenfilename(
        title="Selecione o áudio comprimido para tocar",
        filetypes=[("NumPy Zip", "*.npz")]
    )
    if play_compressed_path:
        lbl_play_input.config(text=os.path.basename(play_compressed_path))
        stop_playback() # Stop any current playback before loading new audio
        
        # Pre-load the decompressed audio data and get its duration
        temp_output = "_temp_play.wav"
        try:
            # A descompressão agora precisa do path completo
            descompressao_audio_PCA(play_compressed_path, temp_output)
            audio_data, audio_sr = sf.read(temp_output, always_2d=True)
            total_duration = len(audio_data) / audio_sr
            slider_time.config(to=total_duration)
            lbl_total_time.config(text=format_time(total_duration))
            slider_time.set(0) # Reset slider to beginning
            os.remove(temp_output) # Clean up temporary file
            enable_player_controls()
        except Exception as e:
            messagebox.showerror("Erro ao carregar áudio", str(e))
            disable_player_controls()

def play_audio_thread_func():
    global is_playing, is_paused, stop_flag, current_position, audio_data, audio_sr, stream

    is_playing = True
    is_paused = False
    stop_flag = False

    if stream is not None:
        stream.close()

    def callback(outdata, frames, time_info, status):
        global current_position, is_playing, stop_flag, is_paused, audio_data, volume_level
        if stop_flag:
            raise sd.CallbackStop
        if is_paused:
            outdata.fill(0)
            return

        chunk_size = frames
        remaining_frames = len(audio_data) - current_position

        if remaining_frames <= 0:
            # End of audio
            is_playing = False
            raise sd.CallbackStop

        if remaining_frames < chunk_size:
            chunk_size = remaining_frames

        # Get the current chunk and apply volume
        current_chunk = audio_data[current_position : current_position + chunk_size] * volume_level
        
        # Ensure the chunk has the correct number of channels
        if current_chunk.shape[1] != outdata.shape[1]:
            # If the audio is mono, duplicate it to fill stereo output
            if current_chunk.shape[1] == 1 and outdata.shape[1] == 2:
                current_chunk = np.hstack((current_chunk, current_chunk))
            else:
                # Handle other channel mismatches if necessary, or raise an error
                print(f"Warning: Channel mismatch. Audio has {current_chunk.shape[1]} but output needs {outdata.shape[1]}.")
                outdata.fill(0) # Fill with silence to avoid errors
                return


        outdata[:chunk_size] = current_chunk
        outdata[chunk_size:] = 0 # Fill remaining with silence if chunk is smaller than frames

        current_position += chunk_size
        root.after(100, update_time_slider) # Update slider periodically

    try:
        stream = sd.OutputStream(samplerate=audio_sr, channels=audio_data.shape[1], callback=callback)
        with stream:
            while is_playing and not stop_flag:
                sd.sleep(100) # Keep thread alive while playing
    except Exception as e:
        print(f"Playback error: {e}")
    finally:
        is_playing = False
        is_paused = False
        stop_flag = False
        current_position = 0
        if stream is not None:
            stream.close()
        # Reset UI
        btn_play.config(text="Play")
        slider_time.set(0)
        lbl_current_time.config(text=format_time(0))

def tocar_comprimido():
    global playback_thread, is_playing, is_paused, stop_flag

    if audio_data is None:
        messagebox.showerror("Erro", "Selecione e carregue um áudio comprimido primeiro!")
        return

    if is_playing and not is_paused: # If currently playing, pause it
        is_paused = True
        btn_play.config(text="Play")
    elif is_paused: # If paused, resume it
        is_paused = False
        btn_play.config(text="Pause")
    else: # If not playing at all, start playback
        stop_flag = False
        is_playing = True
        current_position = 0 # Start from beginning when new playback starts
        btn_play.config(text="Pause")
        
        playback_thread = threading.Thread(target=play_audio_thread_func)
        playback_thread.start()

def pause_playback():
    global is_paused, is_playing
    if is_playing and not is_paused:
        is_paused = True
        btn_play.config(text="Play")
    elif is_paused:
        tocar_comprimido() # Resume playback

def stop_playback():
    global stop_flag, is_playing, is_paused, current_position, stream
    stop_flag = True
    is_playing = False
    is_paused = False
    current_position = 0
    if stream is not None:
        stream.close()
        stream = None
    if playback_thread and playback_thread.is_alive():
        playback_thread.join(timeout=1.0) # Wait for thread to finish
    btn_play.config(text="Play")
    slider_time.set(0)
    lbl_current_time.config(text=format_time(0))

def set_time(val):
    global current_position, audio_sr
    if audio_data is not None:
        new_position = int(float(val) * audio_sr)
        current_position = min(new_position, len(audio_data))
        lbl_current_time.config(text=format_time(float(val)))

def update_time_slider():
    global current_position, audio_sr, total_duration
    if is_playing and not is_paused and audio_data is not None:
        time_elapsed = current_position / audio_sr
        slider_time.set(time_elapsed)
        lbl_current_time.config(text=format_time(time_elapsed))
        if time_elapsed >= total_duration - (1/audio_sr): # Account for floating point
            stop_playback()

def set_volume(val):
    global volume_level
    volume_level = float(val)

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def disable_player_controls():
    btn_play.config(state=tk.DISABLED)
    btn_stop.config(state=tk.DISABLED)
    slider_time.config(state=tk.DISABLED)
    slider_volume.config(state=tk.DISABLED)

def enable_player_controls():
    btn_play.config(state=tk.NORMAL)
    btn_stop.config(state=tk.NORMAL)
    slider_time.config(state=tk.NORMAL)
    slider_volume.config(state=tk.NORMAL)

# ==================== INTERFACE ====================

root = tk.Tk()
root.title("Compressor COCADA")

# Main Title
lbl_main_title = tk.Label(root, text="Compressão Otimizada de Conteúdo Áudio Digital Avançado (C.O.C.A.D.A)", font=("Helvetica", 12, "bold"))
lbl_main_title.pack(pady=10)

# Área compressão
frame_compress = tk.LabelFrame(root, text="Compressão de áudio")
frame_compress.pack(padx=10, pady=5, fill='x')
tk.Button(frame_compress, text="Selecionar áudio para comprimir", command=selecionar_audio_compressao).pack(padx=5, pady=5)
lbl_compressao_input = tk.Label(frame_compress, text="Nenhum arquivo selecionado")
lbl_compressao_input.pack(padx=5, pady=2)

# Opções de qualidade
quality_frame = tk.LabelFrame(frame_compress, text="Qualidade de Compressão")
quality_frame.pack(padx=5, pady=5, fill='x')

quality_var = tk.StringVar(value="media-alta") # Default quality
tk.Radiobutton(quality_frame, text="Alta Qualidade (16-bit)", variable=quality_var, value="alta", command=set_compression_quality).pack(anchor="w", padx=10)
tk.Radiobutton(quality_frame, text="Média-alta Qualidade (16-bit)", variable=quality_var, value="media-alta", command=set_compression_quality).pack(anchor="w", padx=10)
tk.Radiobutton(quality_frame, text="Média-baixa Qualidade (8-bit)", variable=quality_var, value="media-baixa", command=set_compression_quality).pack(anchor="w", padx=10)
tk.Radiobutton(quality_frame, text="Baixa Qualidade (8-bit)", variable=quality_var, value="baixa", command=set_compression_quality).pack(anchor="w", padx=10)

tk.Button(frame_compress, text="Comprimir e salvar (.npz)", command=executar_compressao).pack(padx=5, pady=5)
progress_bar_compress = ttk.Progressbar(frame_compress, orient='horizontal', length=300, mode='determinate')

# Área descompressão
frame_decompress = tk.LabelFrame(root, text="Descompressão de áudio")
frame_decompress.pack(padx=10, pady=5, fill='x')
tk.Button(frame_decompress, text="Selecionar áudio comprimido (.npz)", command=selecionar_comprimido).pack(padx=5, pady=5)
lbl_decompress_input = tk.Label(frame_decompress, text="Nenhum arquivo selecionado")
lbl_decompress_input.pack(padx=5, pady=2)
tk.Button(frame_decompress, text="Descomprimir e salvar (.wav)", command=executar_descompressao).pack(padx=5, pady=5)
progress_bar_decompress = ttk.Progressbar(frame_decompress, orient='horizontal', length=300, mode='determinate')

# Área player
frame_player = tk.LabelFrame(root, text="Player (áudio comprimido)")
frame_player.pack(padx=10, pady=5, fill='x')
tk.Button(frame_player, text="Selecionar áudio comprimido (.npz)", command=selecionar_para_play).pack(padx=5, pady=5)
lbl_play_input = tk.Label(frame_player, text="Nenhum arquivo selecionado")
lbl_play_input.pack(padx=5, pady=2)

# Playback controls
player_controls_frame = tk.Frame(frame_player)
player_controls_frame.pack(pady=5)

btn_play = tk.Button(player_controls_frame, text="Play", command=tocar_comprimido)
btn_play.grid(row=0, column=0, padx=2)
btn_stop = tk.Button(player_controls_frame, text="Stop", command=stop_playback)
btn_stop.grid(row=0, column=2, padx=2)

# Time slider
time_slider_frame = tk.Frame(frame_player)
time_slider_frame.pack(fill='x', padx=5, pady=2)

lbl_current_time = tk.Label(time_slider_frame, text="00:00")
lbl_current_time.pack(side=tk.LEFT)

slider_time = ttk.Scale(time_slider_frame, from_=0, to=100, orient='horizontal', command=set_time, length=300)
slider_time.pack(side=tk.LEFT, expand=True, fill='x', padx=5)

lbl_total_time = tk.Label(time_slider_frame, text="00:00")
lbl_total_time.pack(side=tk.RIGHT)

# Volume slider
volume_frame = tk.Frame(frame_player)
volume_frame.pack(fill='x', padx=5, pady=2)
tk.Label(volume_frame, text="Volume:").pack(side=tk.LEFT)
slider_volume = ttk.Scale(volume_frame, from_=0, to=1, orient='horizontal', command=set_volume)
slider_volume.set(volume_level) # Set initial volume
slider_volume.pack(side=tk.LEFT, expand=True, fill='x', padx=5)

# Disable controls initially
disable_player_controls()

root.mainloop()