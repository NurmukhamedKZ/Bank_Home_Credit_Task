"""
Сервис для автоматического получения резюме из почты.
Поддерживает форматы: PDF, DOCX, и текст в теле письма.
"""

import imaplib
import email
from email.header import decode_header
from email.message import Message
from pathlib import Path
from datetime import datetime
import os
import re
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()


class EmailFetcher:
    """
    Класс для получения резюме из почтового ящика.
    Поддерживает IMAP протокол (Gmail, Yandex, Mail.ru и др.)
    """
    
    # Расширения файлов резюме
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".rtf", ".txt"}
    
    # Ключевые слова для поиска резюме в теме или имени файла
    CV_KEYWORDS = {"cv", "resume", "резюме", "анкета", "кандидат"}
    
    def __init__(
        self,
        email_address: Optional[str] = None,
        email_password: Optional[str] = None,
        imap_server: Optional[str] = None,
        imap_port: int = 993,
        output_dir: Optional[str] = None
    ):
        """
        Инициализация Email Fetcher.
        
        Args:
            email_address: Email адрес (или из EMAIL_ADDRESS в .env)
            email_password: Пароль приложения (или из EMAIL_PASSWORD в .env)
            imap_server: IMAP сервер (или из IMAP_SERVER в .env)
            imap_port: IMAP порт (по умолчанию 993 для SSL)
            output_dir: Папка для сохранения резюме
        """
        self.email_address = email_address or os.getenv("EMAIL_ADDRESS")
        self.email_password = email_password or os.getenv("EMAIL_PASSWORD")
        self.imap_server = imap_server or os.getenv("IMAP_SERVER", "imap.gmail.com")
        self.imap_port = imap_port
        
        # Папка для сохранения резюме
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            project_root = Path(__file__).parent.parent.parent
            self.output_dir = project_root / "data" / "Raw_CVs"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.connection: Optional[imaplib.IMAP4_SSL] = None
    
    def connect(self) -> bool:
        """Подключение к почтовому серверу."""
        if not self.email_address or not self.email_password:
            raise ValueError(
                "Не указаны EMAIL_ADDRESS и EMAIL_PASSWORD. "
                "Добавьте их в .env файл или передайте в конструктор."
            )
        
        try:
            self.connection = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            self.connection.login(self.email_address, self.email_password)
            print(f"✓ Успешно подключено к {self.imap_server}")
            return True
        except imaplib.IMAP4.error as e:
            print(f"✗ Ошибка подключения: {e}")
            return False
    
    def disconnect(self):
        """Отключение от почтового сервера."""
        if self.connection:
            try:
                self.connection.logout()
            except Exception:
                pass
            self.connection = None
    
    def _decode_header_value(self, value: str) -> str:
        """Декодирование заголовка письма."""
        if not value:
            return ""
        
        decoded_parts = []
        for part, encoding in decode_header(value):
            if isinstance(part, bytes):
                decoded_parts.append(part.decode(encoding or "utf-8", errors="ignore"))
            else:
                decoded_parts.append(part)
        return "".join(decoded_parts)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Очистка имени файла от недопустимых символов."""
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', ' ', filename).strip()
        return filename
    
    def _generate_unique_filename(self, filename: str) -> str:
        """Генерация уникального имени файла."""
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            return filename
        
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{timestamp}{ext}"
    
    def _is_cv_file(self, filename: str) -> bool:
        """Проверка, является ли файл резюме по расширению."""
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def _is_cv_related(self, subject: str, filename: str = "") -> bool:
        """Проверка, относится ли письмо/файл к резюме."""
        text = f"{subject} {filename}".lower()
        return any(keyword in text for keyword in self.CV_KEYWORDS)
    
    def _save_attachment(self, part: Message, subject: str) -> Optional[str]:
        """Сохранение вложения из письма."""
        filename = part.get_filename()
        
        if not filename:
            return None
        
        filename = self._decode_header_value(filename)
        filename = self._sanitize_filename(filename)
        
        if not self._is_cv_file(filename):
            return None
        
        payload = part.get_payload(decode=True)
        if not payload:
            return None
        
        unique_filename = self._generate_unique_filename(filename)
        filepath = self.output_dir / unique_filename
        
        with open(filepath, "wb") as f:
            f.write(payload)
        
        print(f"  ✓ Сохранено: {unique_filename}")
        return str(filepath)
    
    def _save_text_cv(self, text: str, subject: str, sender: str) -> Optional[str]:
        """Сохранение текстового резюме из тела письма."""
        if not text or len(text.strip()) < 100:
            return None
        
        safe_subject = self._sanitize_filename(subject)[:50] if subject else "resume"
        safe_sender = self._sanitize_filename(sender.split("@")[0])[:30] if sender else "unknown"
        
        filename = f"{safe_sender}_{safe_subject}.txt"
        unique_filename = self._generate_unique_filename(filename)
        filepath = self.output_dir / unique_filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"От: {sender}\n")
            f.write(f"Тема: {subject}\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n\n")
            f.write(text)
        
        print(f"  ✓ Сохранено текстовое резюме: {unique_filename}")
        return str(filepath)
    
    def fetch_resumes(
        self,
        folder: str = "INBOX",
        search_criteria: str = "UNSEEN",
        save_text_body: bool = True,
        mark_as_read: bool = True
    ) -> List[str]:
        """
        Получение резюме из почтового ящика.
        
        Args:
            folder: Папка почты (INBOX, CV, Resume и т.д.)
            search_criteria: Критерий поиска (UNSEEN, ALL, SINCE "01-Jan-2024")
            save_text_body: Сохранять ли текстовые резюме из тела письма
            mark_as_read: Помечать ли обработанные письма как прочитанные
        
        Returns:
            Список путей к сохраненным файлам
        """
        if not self.connection:
            if not self.connect():
                return []
        
        saved_files = []
        
        try:
            status, _ = self.connection.select(folder)
            if status != "OK":
                print(f"✗ Не удалось открыть папку: {folder}")
                return []
            
            status, messages = self.connection.search(None, search_criteria)
            if status != "OK":
                print("✗ Ошибка поиска писем")
                return []
            
            email_ids = messages[0].split()
            print(f"Найдено писем: {len(email_ids)}")
            
            for email_id in email_ids:
                try:
                    status, msg_data = self.connection.fetch(email_id, "(RFC822)")
                    if status != "OK":
                        continue
                    
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    
                    subject = self._decode_header_value(msg.get("Subject", ""))
                    sender = self._decode_header_value(msg.get("From", ""))
                    
                    print(f"\nОбработка: {subject[:60]}...")
                    
                    has_attachment = False
                    text_body = ""
                    
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition", ""))
                            
                            if "attachment" in content_disposition:
                                filepath = self._save_attachment(part, subject)
                                if filepath:
                                    saved_files.append(filepath)
                                    has_attachment = True
                            elif content_type == "text/plain" and "attachment" not in content_disposition:
                                payload = part.get_payload(decode=True)
                                if payload:
                                    charset = part.get_content_charset() or "utf-8"
                                    text_body += payload.decode(charset, errors="ignore")
                    else:
                        content_type = msg.get_content_type()
                        if content_type == "text/plain":
                            payload = msg.get_payload(decode=True)
                            if payload:
                                charset = msg.get_content_charset() or "utf-8"
                                text_body = payload.decode(charset, errors="ignore")
                    
                    if save_text_body and not has_attachment and text_body:
                        if self._is_cv_related(subject, text_body[:500]):
                            filepath = self._save_text_cv(text_body, subject, sender)
                            if filepath:
                                saved_files.append(filepath)
                    
                    if mark_as_read:
                        self.connection.store(email_id, "+FLAGS", "\\Seen")
                
                except Exception as e:
                    print(f"  ✗ Ошибка обработки письма: {e}")
                    continue
        
        except Exception as e:
            print(f"✗ Ошибка: {e}")
        
        print(f"\n{'='*50}")
        print(f"Всего сохранено файлов: {len(saved_files)}")
        
        return saved_files
    
    def __enter__(self):
        """Context manager вход."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager выход."""
        self.disconnect()
