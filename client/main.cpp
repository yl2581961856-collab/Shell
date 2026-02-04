#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>

#include <boost/asio/steady_timer.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using websocketpp::connection_hdl;

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

static void usage() {
    std::cout << "Usage: asr_ws_client --uri ws://127.0.0.1:8000/asr/stream "
                 "[--file audio.pcm] [--chunk-bytes 3200] [--bps 32000] "
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
    std::string session_id;
    int sample_rate = 16000;
    int channels = 1;
    std::string fmt = "pcm_s16le";
    std::string encoding = "pcm";
    size_t chunk_bytes = 3200;
    size_t bytes_per_sec = 32000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--uri" && i + 1 < argc) {
            uri = argv[++i];
        } else if (arg == "--file" && i + 1 < argc) {
            file = argv[++i];
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
        } else if (arg == "--chunk-bytes" && i + 1 < argc) {
            chunk_bytes = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--bps" && i + 1 < argc) {
            bytes_per_sec = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "-h" || arg == "--help") {
            usage();
            return 0;
        } else {
            usage();
            return 1;
        }
    }

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
        if (!file.empty()) {
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

        std::cout << "Press Enter to exit..." << std::endl;
        std::string line;
        std::getline(std::cin, line);

        if (!sent_end) {
            client.send_text("{\"type\":\"end\",\"session_id\":\"" + session_id + "\"}");
        }
        client.stop();
        io_thread.join();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }
}
