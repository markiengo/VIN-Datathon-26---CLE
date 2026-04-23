CREATE TABLE `customers` (
  `customer_id` BIGINT NOT NULL,
  `zip` BIGINT NULL,
  `city` VARCHAR(100) NULL,
  `signup_date` DATE NULL,
  `gender` VARCHAR(20) NULL,
  `age_group` VARCHAR(50) NULL,
  `acquisition_channel` VARCHAR(100) NULL,
  PRIMARY KEY (`customer_id`)
);

CREATE TABLE `geography` (
  `zip` BIGINT NOT NULL,
  `city` VARCHAR(100) NULL,
  `region` VARCHAR(100) NULL,
  `district` VARCHAR(100) NULL,
  PRIMARY KEY (`zip`)
);

CREATE TABLE `orders` (
  `order_id` BIGINT NOT NULL,
  `customer_id` BIGINT NULL,
  `zip` BIGINT NULL,
  `order_date` DATE NULL,
  `order_status` VARCHAR(50) NULL,
  `payment_method` VARCHAR(50) NULL,
  `device_type` VARCHAR(50) NULL,
  `order_source` VARCHAR(100) NULL,
  PRIMARY KEY (`order_id`),
  FOREIGN KEY (`customer_id`) REFERENCES `customers` (`customer_id`),
  FOREIGN KEY (`zip`) REFERENCES `geography` (`zip`)
);

CREATE TABLE `products` (
  `product_id` BIGINT NOT NULL,
  `product_name` VARCHAR(255) NULL,
  `category` VARCHAR(100) NULL,
  `segment` VARCHAR(100) NULL,
  `size` VARCHAR(20) NULL,
  `color` VARCHAR(50) NULL,
  `price` DECIMAL(12,2) NULL,
  `cogs` DECIMAL(12,2) NULL,
  PRIMARY KEY (`product_id`)
);

CREATE TABLE `promotions` (
  `promo_id` VARCHAR(50) NOT NULL,
  `promo_name` VARCHAR(255) NULL,
  `promo_type` VARCHAR(100) NULL,
  `discount_value` DECIMAL(12,2) NULL,
  `start_date` DATE NULL,
  `end_date` DATE NULL,
  `applicable_category` VARCHAR(100) NULL,
  `promo_channel` VARCHAR(100) NULL,
  `stackable_flag` TINYINT NULL,
  `min_order_value` DOUBLE NULL,
  PRIMARY KEY (`promo_id`)
);

CREATE TABLE `order_items` (
  `order_id` BIGINT NOT NULL,
  `product_id` BIGINT NOT NULL,
  `quantity` BIGINT NULL,
  `unit_price` DECIMAL(12,2) NULL,
  `discount_amount` DECIMAL(12,2) NULL,
  `promo_id` VARCHAR(50) NULL,
  `promo_id_2` VARCHAR(50) NULL,
  FOREIGN KEY (`order_id`) REFERENCES `orders` (`order_id`),
  FOREIGN KEY (`product_id`) REFERENCES `products` (`product_id`),
  FOREIGN KEY (`promo_id`) REFERENCES `promotions` (`promo_id`)
);

CREATE TABLE `payments` (
  `order_id` BIGINT NOT NULL,
  `payment_method` VARCHAR(50) NULL,
  `payment_value` DECIMAL(12,2) NULL,
  `installments` BIGINT NULL,
  PRIMARY KEY (`order_id`),
  FOREIGN KEY (`order_id`) REFERENCES `orders` (`order_id`)
);

CREATE TABLE `shipments` (
  `order_id` BIGINT NOT NULL,
  `ship_date` DATE NULL,
  `delivery_date` DATE NULL,
  `shipping_fee` DECIMAL(12,2) NULL,
  PRIMARY KEY (`order_id`),
  FOREIGN KEY (`order_id`) REFERENCES `orders` (`order_id`)
);

CREATE TABLE `returns` (
  `return_id` VARCHAR(50) NOT NULL,
  `order_id` BIGINT NULL,
  `product_id` BIGINT NULL,
  `return_date` DATE NULL,
  `return_reason` VARCHAR(100) NULL,
  `return_quantity` BIGINT NULL,
  `refund_amount` DECIMAL(12,2) NULL,
  PRIMARY KEY (`return_id`),
  FOREIGN KEY (`order_id`) REFERENCES `orders` (`order_id`)
);

CREATE TABLE `reviews` (
  `review_id` VARCHAR(50) NOT NULL,
  `order_id` BIGINT NULL,
  `product_id` BIGINT NULL,
  `customer_id` BIGINT NULL,
  `review_date` DATE NULL,
  `rating` BIGINT NULL,
  `review_title` VARCHAR(255) NULL,
  PRIMARY KEY (`review_id`),
  FOREIGN KEY (`order_id`) REFERENCES `orders` (`order_id`)
);

CREATE TABLE `inventory` (
  `snapshot_date` DATE NOT NULL,
  `product_id` BIGINT NOT NULL,
  `stock_on_hand` BIGINT NULL,
  `units_received` BIGINT NULL,
  `units_sold` BIGINT NULL,
  `stockout_days` BIGINT NULL,
  `days_of_supply` DECIMAL(12,2) NULL,
  `fill_rate` DECIMAL(8,4) NULL,
  `stockout_flag` TINYINT NULL,
  `overstock_flag` TINYINT NULL,
  `reorder_flag` TINYINT NULL,
  `sell_through_rate` DECIMAL(8,4) NULL,
  `year` BIGINT NULL,
  `month` BIGINT NULL,
  PRIMARY KEY (`snapshot_date`, `product_id`),
  FOREIGN KEY (`product_id`) REFERENCES `products` (`product_id`)
);

CREATE TABLE `sales` (
  `Date` DATE NOT NULL,
  `Revenue` DECIMAL(14,2) NULL,
  `COGS` DECIMAL(14,2) NULL
);

CREATE TABLE `web_traffic` (
  `date` DATE NOT NULL,
  `sessions` BIGINT NULL,
  `unique_visitors` BIGINT NULL,
  `page_views` BIGINT NULL,
  `bounce_rate` DECIMAL(8,4) NULL,
  `avg_session_duration_sec` DECIMAL(12,2) NULL,
  `traffic_source` VARCHAR(100) NULL
);
