#!/usr/bin/env python3
"""
Pushover Notification System for Jarvis
Sends notifications about user engagement and system events
"""

import os
import logging
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PushoverNotifier:
    """Handles Pushover notifications for Jarvis events"""
    
    def __init__(self):
        self.api_token = os.getenv('PUSHOVER_API_TOKEN')
        self.user_key = os.getenv('PUSHOVER_USER_KEY')
        self.enabled = os.getenv('PUSHOVER_ENABLED', 'false').lower() == 'true'
        self.api_url = "https://api.pushover.net/1/messages.json"
        
        # Rate limiting to avoid spam
        self.last_new_user_notification = {}  # Track notifications per user
        self.notification_cooldown = timedelta(hours=1)  # Only notify once per hour per user
        
        if self.enabled and (not self.api_token or not self.user_key):
            logger.warning("Pushover enabled but missing API token or user key")
            self.enabled = False
        elif self.enabled:
            logger.info("Pushover notifications enabled")
        else:
            logger.debug("Pushover notifications disabled")
    
    async def send_notification(self, message: str, title: str = "Jarvis Alert", 
                              priority: int = 0, sound: Optional[str] = None) -> bool:
        """
        Send a Pushover notification
        
        Args:
            message: The notification message
            title: The notification title
            priority: Priority level (-2 to 2, 0 = normal)
            sound: Sound name (optional)
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Pushover disabled, would send: {title}: {message}")
            return False
        
        try:
            data = {
                "token": self.api_token,
                "user": self.user_key,
                "message": message,
                "title": title,
                "priority": priority,
                "timestamp": int(datetime.now().timestamp())
            }
            
            if sound:
                data["sound"] = sound
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, data=data) as response:
                    if response.status == 200:
                        logger.info(f"Pushover notification sent: {title}")
                        return True
                    else:
                        logger.error(f"Pushover API error {response.status}: {await response.text()}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Pushover notification: {e}")
            return False
    
    async def notify_new_user_conversation(self, user_info: Dict[str, Any], 
                                         message_preview: str, request_ip: str) -> bool:
        """
        Notify about a new user starting a conversation with Jarvis
        
        Args:
            user_info: Information about the user (from auth system)
            message_preview: First few words of their message
            request_ip: IP address of the request
        
        Returns:
            bool: True if notification sent, False if skipped or failed
        """
        if not self.enabled:
            return False
        
        user_identifier = user_info.get('name', 'Unknown User')
        
        # Check rate limiting - only notify once per user per cooldown period
        now = datetime.now()
        last_notification = self.last_new_user_notification.get(user_identifier)
        
        if last_notification and (now - last_notification) < self.notification_cooldown:
            logger.debug(f"Skipping notification for {user_identifier} - rate limited")
            return False
        
        # Update rate limiting tracker
        self.last_new_user_notification[user_identifier] = now
        
        # Create notification message
        preview = message_preview[:50] + "..." if len(message_preview) > 50 else message_preview
        
        message = f"""New conversation started with Jarvis!
        
User: {user_identifier}
IP: {request_ip}
Message: "{preview}"
Time: {now.strftime('%Y-%m-%d %H:%M:%S')}"""
        
        title = "🤖 New Jarvis User"
        
        return await self.send_notification(
            message=message,
            title=title,
            priority=0,  # Normal priority
            sound="intermission"  # Gentle notification sound
        )
    
    async def notify_system_event(self, event_type: str, details: str, 
                                priority: int = 0) -> bool:
        """
        Notify about system events (errors, restarts, etc.)
        
        Args:
            event_type: Type of event (e.g., "API_START", "ERROR", "RESTART")
            details: Event details
            priority: Notification priority
        
        Returns:
            bool: True if notification sent, False otherwise
        """
        if not self.enabled:
            return False
        
        message = f"Jarvis System Event: {event_type}\n\n{details}"
        title = f"🔧 Jarvis {event_type}"
        
        return await self.send_notification(
            message=message,
            title=title,
            priority=priority,
            sound="mechanical" if priority > 0 else None
        )
    
    async def notify_usage_milestone(self, milestone_type: str, count: int) -> bool:
        """
        Notify about usage milestones (e.g., 100th conversation)
        
        Args:
            milestone_type: Type of milestone (e.g., "conversations", "users")
            count: The milestone number
        
        Returns:
            bool: True if notification sent, False otherwise
        """
        if not self.enabled:
            return False
        
        message = f"Jarvis has reached {count} {milestone_type}! 🎉"
        title = f"🏆 Jarvis Milestone"
        
        return await self.send_notification(
            message=message,
            title=title,
            priority=0,
            sound="magic"
        )
    
    def is_enabled(self) -> bool:
        """Check if notifications are enabled and configured"""
        return self.enabled

# Global notifier instance
_notifier = None

def get_notifier() -> PushoverNotifier:
    """Get the global notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = PushoverNotifier()
    return _notifier

# Convenience functions
async def notify_new_user(user_info: Dict[str, Any], message_preview: str, 
                         request_ip: str) -> bool:
    """Convenience function to notify about new user conversations"""
    notifier = get_notifier()
    return await notifier.notify_new_user_conversation(user_info, message_preview, request_ip)

async def notify_system_event(event_type: str, details: str, priority: int = 0) -> bool:
    """Convenience function to notify about system events"""
    notifier = get_notifier()
    return await notifier.notify_system_event(event_type, details, priority)

async def notify_milestone(milestone_type: str, count: int) -> bool:
    """Convenience function to notify about milestones"""
    notifier = get_notifier()
    return await notifier.notify_usage_milestone(milestone_type, count)