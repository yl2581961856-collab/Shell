#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>

#include <boost/asio/steady_timer.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <csignal>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(ASRWS_ENABLE_PORTAUDIO)
#include <portaudio.h>
#endif

using websocketpp::connection_hdl;

static std::atomic<bool> g_stop{false};

#ifdef _WIN32
static BOOL WINAPI console_ctrl_handler(DWORD type) {
    switch (type) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            g_stop.store(true);
            return TRUE;
        default:
            return FALSE;
    }
}

static void install_signal_handlers() {
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
}
#else
static void posix_signal_handler(int) {
    g_stop.store(true);
}

static void install_signal_handlers() {
    std::signal(SIGINT, posix_signal_handler);
    std::signal(SIGTERM, posix_signal_handler);
}
#endif

class WsClient {
public:
    using Client = websocketpp::client<websocketpp::config::asio_client>;

    WsClient() : open_(false) {
        client_.clear_access_channels(websocketpp::log::alevel::all);
        client_.clear_error_channels(websocketpp::log::elevel::all);
        client_.init_asio();

        client_.set_open_handler([this](connection_hdl hdl) {
            hdl_ = hdl;
            open_.store(true);
            start_ping();
            log("connected");
        });

        client_.set_fail_handler([this](connection_hdl) {
            open_.store(false);
            log("connection failed");
        });

        client_.set_close_handler([this](connection_hdl) {
            open_.store(false);
            log("connection closed");
        });

        client_.set_message_handler([this](connection_hdl, Client::message_ptr msg) {
            log(std::string("recv: ") + msg->get_payload());
        });

        client_.set_pong_handler([this](connection_hdl, std::string) {
            log("pong");
            return true;
        });
    }

    void connect(const std::string& uri) {
        websocketpp::lib::error_code ec;
        Client::connection_ptr con = client_.get_connection(uri, ec);
        if (ec) {
            throw std::runtime_error(std::string("get_connection failed: ") + ec.message());
        }
        client_.connect(con);
    }

    void run() {
        client_.run();
    }

    void stop() {
        client_.stop();
    }

    bool is_open() const {
        return open_.load();
    }

    void send_binary(std::vector<uint8_t> data) {
        if (!open_.load()) {
            return;
        }
        client_.get_io_service().post([this, data = std::move(data)]() {
            websocketpp::lib::error_code ec;
            client_.send(hdl_, data.data(), data.size(), websocketpp::frame::opcode::binary, ec);
            if (ec) {
                log(std::string("send failed: ") + ec.message());
            }
        });
    }

    void send_text(std::string text) {
        if (!open_.load()) {
            return;
        }
        client_.get_io_service().post([this, text = std::move(text)]() {
            websocketpp::lib::error_code ec;
            client_.send(hdl_, text, websocketpp::frame::opcode::text, ec);
            if (ec) {
                log(std::string("send failed: ") + ec.message());
            }
        });
    }

private:
    void start_ping() {
        auto timer = std::make_shared<boost::asio::steady_timer>(client_.get_io_service());
        ping_timer_ = timer;
        schedule_ping();
    }

    void schedule_ping() {
        if (!open_.load()) {
            return;
        }
        ping_timer_->expires_after(std::chrono::seconds(20));
        ping_timer_->async_wait([this](const boost::system::error_code& ec) {
            if (ec || !open_.load()) {
                return;
            }
            websocketpp::lib::error_code ping_ec;
            client_.ping(hdl_, "", ping_ec);
            if (ping_ec) {
                log(std::string("ping failed: ") + ping_ec.message());
            }
            schedule_ping();
        });
    }

    void log(const std::string& msg) {
        std::lock_guard<std::mutex> lock(log_mutex_);
        std::cout << msg << std::endl;
    }

    Client client_;
    connection_hdl hdl_;
    std::atomic<bool> open_;
    std::shared_ptr<boost::asio::steady_timer> ping_timer_;
    std::mutex log_mutex_;
};

#if defined(ASRWS_ENABLE_PORTAUDIO)
static void pa_check(PaError err, const char* what) {
    if (err == paNoError || err == paInputOverflowed) {
        return;
    }
    throw std::runtime_error(std::string(what) + ": " + Pa_GetErrorText(err));
}

class PortAudioGuard {
public:
    PortAudioGuard() {
        pa_check(Pa_Initialize(), "Pa_Initialize");
    }

    ~PortAudioGuard() {
        Pa_Terminate();
    }

    PortAudioGuard(const PortAudioGuard&) = delete;
    PortAudioGuard& operator=(const PortAudioGuard&) = delete;
};

static void list_pa_input_devices() {
    PortAudioGuard guard;
    const int num = Pa_GetDeviceCount();
    const PaDeviceIndex def_in = Pa_GetDefaultInputDevice();

    std::cout << "PortAudio input devices:" << std::endl;
    for (int i = 0; i < num; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (!info || info->maxInputChannels <= 0) {
            continue;
        }
        std::cout << (i == def_in ? "* " : "  ") << i << ": " << info->name << " (max_ch="
                  << info->maxInputChannels << ", default_sr=" << info->defaultSampleRate << ")"
                  << std::endl;
    }
}

class MicInputStream {
public:
    MicInputStream(int device_index, int sample_rate, int channels, unsigned long frames_per_buffer)
        : frames_per_buffer_(frames_per_buffer), channels_(channels) {
        input_params_.device = (device_index >= 0) ? device_index : Pa_GetDefaultInputDevice();
        if (input_params_.device == paNoDevice) {
            throw std::runtime_error("No default input device");
        }

        const PaDeviceInfo* info = Pa_GetDeviceInfo(input_params_.device);
        if (!info || info->maxInputChannels < channels) {
            throw std::runtime_error("Input device does not support requested channels");
        }

        input_params_.channelCount = channels;
        input_params_.sampleFormat = paInt16;
        input_params_.suggestedLatency = info->defaultLowInputLatency;
        input_params_.hostApiSpecificStreamInfo = nullptr;

        pa_check(Pa_OpenStream(&stream_, &input_params_, nullptr, sample_rate, frames_per_buffer_, paClipOff, nullptr,
                               nullptr),
                 "Pa_OpenStream");
        pa_check(Pa_StartStream(stream_), "Pa_StartStream");

        buf_.resize(static_cast<size_t>(frames_per_buffer_) * static_cast<size_t>(channels_));
    }

    ~MicInputStream() {
        if (stream_) {
            Pa_AbortStream(stream_);
            Pa_CloseStream(stream_);
            stream_ = nullptr;
        }
    }

    MicInputStream(const MicInputStream&) = delete;
    MicInputStream& operator=(const MicInputStream&) = delete;

    std::vector<uint8_t> read_bytes() {
        pa_check(Pa_ReadStream(stream_, buf_.data(), frames_per_buffer_), "Pa_ReadStream");
        const size_t bytes = buf_.size() * sizeof(int16_t);
        std::vector<uint8_t> out(bytes);
        std::memcpy(out.data(), buf_.data(), bytes);
        return out;
    }

private:
    PaStream* stream_ = nullptr;
    PaStreamParameters input_params_{};
    unsigned long frames_per_buffer_;
    int channels_;
    std::vector<int16_t> buf_;
};
#endif

static void usage() {
    std::cout << "Usage: asr_ws_client --uri ws://127.0.0.1:8000/asr/stream "
                 "[--file audio.pcm] [--chunk-bytes 3200] [--bps 32000] "
                 "[--mic] [--chunk-ms 100] [--duration 0] [--device N] [--list-devices] "
                 "[--session-id id] [--sr 16000] [--ch 1] [--fmt pcm_s16le] [--encoding pcm]"
              << std::endl;
}

static std::vector<uint8_t> read_chunk(std::ifstream& in, size_t bytes) {
    std::vector<uint8_t> buf(bytes);
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(bytes));
    std::streamsize got = in.gcount();
    buf.resize(static_cast<size_t>(got));
    return buf;
}

int main(int argc, char** argv) {
    std::string uri = "ws://127.0.0.1:8000/asr/stream";
    std::string file;
    bool use_mic = false;
    bool list_devices = false;
    int mic_device = -1;
    std::string session_id;
    int sample_rate = 16000;
    int channels = 1;
    std::string fmt = "pcm_s16le";
    std::string encoding = "pcm";
    int chunk_ms = 100;
    double duration_sec = 0.0;
    size_t chunk_bytes = 0;
    size_t bytes_per_sec = 0;
    bool chunk_bytes_set = false;
    bool bps_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--uri" && i + 1 < argc) {
            uri = argv[++i];
        } else if (arg == "--file" && i + 1 < argc) {
            file = argv[++i];
        } else if (arg == "--mic") {
            use_mic = true;
        } else if (arg == "--list-devices") {
            list_devices = true;
        } else if (arg == "--device" && i + 1 < argc) {
            mic_device = std::stoi(argv[++i]);
        } else if (arg == "--session-id" && i + 1 < argc) {
            session_id = argv[++i];
        } else if (arg == "--sr" && i + 1 < argc) {
            sample_rate = std::stoi(argv[++i]);
        } else if (arg == "--ch" && i + 1 < argc) {
            channels = std::stoi(argv[++i]);
        } else if (arg == "--fmt" && i + 1 < argc) {
            fmt = argv[++i];
        } else if (arg == "--encoding" && i + 1 < argc) {
            encoding = argv[++i];
        } else if (arg == "--chunk-ms" && i + 1 < argc) {
            chunk_ms = std::stoi(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            duration_sec = std::stod(argv[++i]);
        } else if (arg == "--chunk-bytes" && i + 1 < argc) {
            chunk_bytes = static_cast<size_t>(std::stoul(argv[++i]));
            chunk_bytes_set = true;
        } else if (arg == "--bps" && i + 1 < argc) {
            bytes_per_sec = static_cast<size_t>(std::stoul(argv[++i]));
            bps_set = true;
        } else if (arg == "-h" || arg == "--help") {
            usage();
            return 0;
        } else {
            usage();
            return 1;
        }
    }

    if (use_mic && !file.empty()) {
        std::cerr << "Error: --mic and --file are mutually exclusive" << std::endl;
        return 1;
    }

    if (chunk_ms <= 0) {
        std::cerr << "Error: --chunk-ms must be > 0" << std::endl;
        return 1;
    }

    const size_t bytes_per_sample = 2; // pcm_s16le
    if (!bps_set) {
        bytes_per_sec = static_cast<size_t>(sample_rate) * static_cast<size_t>(channels) * bytes_per_sample;
    }
    if (!chunk_bytes_set) {
        chunk_bytes = (static_cast<size_t>(sample_rate) * static_cast<size_t>(channels) * bytes_per_sample *
                       static_cast<size_t>(chunk_ms)) /
                      1000;
    }
    if (bytes_per_sec == 0 || chunk_bytes == 0) {
        std::cerr << "Error: invalid audio parameters (sr/ch/chunk-ms/bps)" << std::endl;
        return 1;
    }

    if (list_devices) {
#if defined(ASRWS_ENABLE_PORTAUDIO)
        try {
            list_pa_input_devices();
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Failed to list devices: " << e.what() << std::endl;
            return 1;
        }
#else
        std::cerr << "Error: --list-devices requires PortAudio. Rebuild with -DASRWS_ENABLE_PORTAUDIO=ON"
                  << std::endl;
        return 1;
#endif
    }

#if !defined(ASRWS_ENABLE_PORTAUDIO)
    if (use_mic) {
        std::cerr << "Error: --mic requires PortAudio. Rebuild with -DASRWS_ENABLE_PORTAUDIO=ON" << std::endl;
        return 1;
    }
#endif

    try {
        WsClient client;
        client.connect(uri);

        std::thread io_thread([&client]() { client.run(); });

        const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!client.is_open() && std::chrono::steady_clock::now() < wait_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!client.is_open()) {
            std::cerr << "Failed to open websocket within timeout" << std::endl;
            client.stop();
            io_thread.join();
            return 1;
        }

        if (session_id.empty()) {
            const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();
            session_id = "cpp-" + std::to_string(now_ms);
        }
        std::ostringstream start_frame;
        start_frame << "{\"type\":\"start\",\"sr\":" << sample_rate << ",\"fmt\":\"" << fmt
                    << "\",\"ch\":" << channels << ",\"encoding\":\"" << encoding
                    << "\",\"session_id\":\"" << session_id << "\"}";
        client.send_text(start_frame.str());

        bool sent_end = false;
        if (use_mic) {
#if defined(ASRWS_ENABLE_PORTAUDIO)
            g_stop.store(false);
            install_signal_handlers();

            PortAudioGuard pa_guard;
            const unsigned long chunk_frames =
                static_cast<unsigned long>((static_cast<long long>(sample_rate) * static_cast<long long>(chunk_ms)) /
                                           1000LL);
            if (chunk_frames == 0) {
                throw std::runtime_error("chunk size too small (chunk_frames==0)");
            }

            std::cout << "Streaming microphone (" << sample_rate << " Hz, ch=" << channels << ", chunk_ms="
                      << chunk_ms << "). ";
            if (duration_sec > 0.0) {
                std::cout << "Will stop after " << duration_sec << " sec." << std::endl;
            } else {
                std::cout << "Press Ctrl+C to stop." << std::endl;
            }

            MicInputStream mic(mic_device, sample_rate, channels, chunk_frames);

            const auto t0 = std::chrono::steady_clock::now();
            while (client.is_open() && !g_stop.load()) {
                if (duration_sec > 0.0) {
                    const auto elapsed =
                        std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() -
                                                                                  t0)
                            .count();
                    if (elapsed >= duration_sec) {
                        break;
                    }
                }

                std::vector<uint8_t> chunk = mic.read_bytes();
                if (!chunk.empty()) {
                    client.send_binary(std::move(chunk));
                }
            }

            client.send_text("{\"type\":\"end\",\"session_id\":\"" + session_id + "\"}");
            sent_end = true;
#else
            std::cerr << "Error: --mic requires PortAudio. Rebuild with -DASRWS_ENABLE_PORTAUDIO=ON" << std::endl;
            client.send_text("{\"type\":\"end\",\"session_id\":\"" + session_id + "\"}");
            client.stop();
            io_thread.join();
            return 1;
#endif
        } else if (!file.empty()) {
            std::ifstream in(file, std::ios::binary);
            if (!in) {
                std::cerr << "Failed to open file: " << file << std::endl;
                client.send_text("{\"type\":\"end\",\"session_id\":\"" + session_id + "\"}");
                client.stop();
                io_thread.join();
                return 1;
            }

            const auto sleep_ms = static_cast<int>(
                (static_cast<double>(chunk_bytes) / static_cast<double>(bytes_per_sec)) * 1000.0);

            while (in && client.is_open()) {
                std::vector<uint8_t> chunk = read_chunk(in, chunk_bytes);
                if (chunk.empty()) {
                    break;
                }
                client.send_binary(std::move(chunk));
                if (sleep_ms > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
                }
            }
            client.send_text("{\"type\":\"end\",\"session_id\":\"" + session_id + "\"}");
            sent_end = true;
        }

        if (!use_mic) {
            std::cout << "Press Enter to exit..." << std::endl;
            std::string line;
            std::getline(std::cin, line);
        }

        if (!sent_end) {
            client.send_text("{\"type\":\"end\",\"session_id\":\"" + session_id + "\"}");
        }

        // Give the server a moment to respond with final/close after `end` (especially in mic mode).
        const auto close_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (client.is_open() && std::chrono::steady_clock::now() < close_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        client.stop();
        io_thread.join();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }
}
